#  Copyright (c) 2023 Macula
#    macula.dev, China
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# @File : model_worker.py
# @author: rain
# @Date：2023/11/18

import multiprocessing as mp
import os
import sys

from fastapi import FastAPI

from configs import (
    LLM_MODELS,
    LOG_PATH,
    MODEL_PATH
)
from server.utils import get_model_worker_config, fschat_model_worker_address, fschat_controller_address, set_app_event


def create_model_worker_app(log_level: str = "INFO", **kwargs) -> FastAPI:
    """
    kwargs包含的字段如下：
    host:
    port:
    model_names:[`model_name`]
    controller_address:
    worker_address:

    对于Langchain支持的模型：
        langchain_model:True
        不会使用fschat
    对于online_api:
        online_api:True
        worker_class: `provider`
    对于离线模型：
        model_path: `model_name_or_path`,huggingface的repo-id或本地路径
        device:`LLM_DEVICE`
    """
    import fastchat.constants
    import argparse
    from fastchat.serve.model_worker import worker_id
    
    fastchat.constants.LOGDIR = LOG_PATH

    parser = argparse.ArgumentParser()
    args = parser.parse_args([])

    for k, v in kwargs.items():
        setattr(args, k, v)

    # 在线模型API
    if worker_class := kwargs.get("worker_class"):
        from fastchat.serve.base_model_worker import app

        worker = worker_class(model_names=args.model_names,
                              controller_addr=args.controller_address,
                              worker_addr=args.worker_address)
        # sys.modules["fastchat.serve.base_model_worker"].worker = worker
        sys.modules["fastchat.serve.base_model_worker"].logger.setLevel(log_level)
    # 本地模型
    else:
        from configs.model_config import VLLM_MODEL_DICT
        if kwargs["model_names"][0] in VLLM_MODEL_DICT and args.infer_turbo == "vllm":
            import fastchat.serve.vllm_worker
            from fastchat.serve.vllm_worker import VLLMWorker, app
            from vllm import AsyncLLMEngine
            from vllm.engine.arg_utils import AsyncEngineArgs

            args.tokenizer = args.model_path  # 如果tokenizer与model_path不一致在此处添加
            args.tokenizer_mode = 'auto'
            args.trust_remote_code = True
            args.download_dir = None
            args.load_format = 'auto'
            args.dtype = 'auto'
            args.seed = 0
            args.worker_use_ray = False
            args.pipeline_parallel_size = 1
            args.tensor_parallel_size = 1
            args.block_size = 16
            args.swap_space = 4  # GiB
            args.gpu_memory_utilization = 0.90
            args.max_num_batched_tokens = None  # 一个批次中的最大令牌（tokens）数量，这个取决于你的显卡和大模型设置，设置太大显存会不够
            args.max_num_seqs = 256
            args.disable_log_stats = False
            args.conv_template = None
            args.limit_worker_concurrency = 5
            args.no_register = False
            args.num_gpus = 4  # vllm worker的切分是tensor并行，这里填写显卡的数量
            args.engine_use_ray = False
            args.disable_log_requests = False

            # 0.2.1 vllm后要加的参数, 但是这里不需要
            args.max_model_len = None
            args.revision = None
            args.quantization = None
            args.max_log_len = None
            args.tokenizer_revision = None

            if args.model_path:
                args.model = args.model_path
            if args.num_gpus > 1:
                args.tensor_parallel_size = args.num_gpus

            for k, v in kwargs.items():
                setattr(args, k, v)

            engine_args = AsyncEngineArgs.from_cli_args(args)
            engine = AsyncLLMEngine.from_engine_args(engine_args)

            worker = VLLMWorker(
                controller_addr=args.controller_address,
                worker_addr=args.worker_address,
                worker_id=worker_id,
                model_path=args.model_path,
                model_names=args.model_names,
                limit_worker_concurrency=args.limit_worker_concurrency,
                no_register=args.no_register,
                llm_engine=engine,
                conv_template=args.conv_template,
            )
            sys.modules["fastchat.serve.vllm_worker"].engine = engine
            # sys.modules["fastchat.serve.vllm_worker"].worker = worker
            sys.modules["fastchat.serve.vllm_worker"].logger.setLevel(log_level)

        elif kwargs["model_names"][0] in MODEL_PATH["llm_model"]:
            from fastchat.serve.model_worker import app, GptqConfig, AWQConfig, ModelWorker

            args.gpus = "0"  # GPU的编号,如果有多个GPU，可以设置为"0,1,2,3"
            args.max_gpu_memory = "22GiB"
            args.num_gpus = 1  # model worker的切分是model并行，这里填写显卡的数量

            args.load_8bit = False
            args.cpu_offloading = None
            args.gptq_ckpt = None
            args.gptq_wbits = 16
            args.gptq_groupsize = -1
            args.gptq_act_order = False
            args.awq_ckpt = None
            args.awq_wbits = 16
            args.awq_groupsize = -1
            args.model_names = [""]
            args.conv_template = None
            args.limit_worker_concurrency = 5
            args.stream_interval = 2
            args.no_register = False
            args.embed_in_truncate = False
            for k, v in kwargs.items():
                setattr(args, k, v)
            if args.gpus:
                if args.num_gpus is None:
                    args.num_gpus = len(args.gpus.split(','))
                if len(args.gpus.split(",")) < args.num_gpus:
                    raise ValueError(
                        f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
                    )
                os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
            gptq_config = GptqConfig(
                ckpt=args.gptq_ckpt or args.model_path,
                wbits=args.gptq_wbits,
                groupsize=args.gptq_groupsize,
                act_order=args.gptq_act_order,
            )
            awq_config = AWQConfig(
                ckpt=args.awq_ckpt or args.model_path,
                wbits=args.awq_wbits,
                groupsize=args.awq_groupsize,
            )

            worker = ModelWorker(
                controller_addr=args.controller_address,
                worker_addr=args.worker_address,
                worker_id=worker_id,
                model_path=args.model_path,
                model_names=args.model_names,
                limit_worker_concurrency=args.limit_worker_concurrency,
                no_register=args.no_register,
                device=args.device,
                num_gpus=args.num_gpus,
                max_gpu_memory=args.max_gpu_memory,
                load_8bit=args.load_8bit,
                cpu_offloading=args.cpu_offloading,
                gptq_config=gptq_config,
                awq_config=awq_config,
                stream_interval=args.stream_interval,
                conv_template=args.conv_template,
                embed_in_truncate=args.embed_in_truncate,
            )
            sys.modules["fastchat.serve.model_worker"].args = args
            sys.modules["fastchat.serve.model_worker"].gptq_config = gptq_config
            # sys.modules["fastchat.serve.model_worker"].worker = worker
            sys.modules["fastchat.serve.model_worker"].logger.setLevel(log_level)

    # MakeFastAPIOffline(app)
    app.title = f"FastChat LLM Server ({args.model_names[0]})"
    app._worker = worker
    return app


def run_model_worker(
        model_name: str = LLM_MODELS[0],
        controller_address: str = "",
        log_level: str = "INFO",
        q: mp.Queue = None,
        started_event: mp.Event = None,
):
    import uvicorn
    import sys
    from server.utils import set_httpx_config

    set_httpx_config()

    kwargs = get_model_worker_config(model_name)
    host = kwargs.pop("host")
    port = kwargs.pop("port")
    kwargs["model_names"] = [model_name]
    kwargs["controller_address"] = controller_address or fschat_controller_address()
    kwargs["worker_address"] = fschat_model_worker_address(model_name)
    model_path = kwargs.get("model_path", "")
    kwargs["model_path"] = model_path

    app = create_model_worker_app(log_level=log_level, **kwargs)
    set_app_event(app, started_event)
    if log_level == "ERROR":
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())
