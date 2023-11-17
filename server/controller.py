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

# @File : controller.py
# @author: rain
# @Date：2023/11/18

import multiprocessing as mp
import sys

from fastapi import FastAPI

from configs import (
    LOG_PATH,
    FSCHAT_CONTROLLER
)
from server.utils import set_app_event


def create_controller_app(
        dispatch_method: str,
        log_level: str = "INFO",
) -> FastAPI:
    """
    创建FastChat的Controller
    :param dispatch_method: model worker的负载均衡方式
    :param log_level: 日志等级
    :return: FastAPI server
    """
    import fastchat.constants
    from fastchat.serve.controller import app, Controller, logger

    fastchat.constants.LOGDIR = LOG_PATH
    logger.setLevel(log_level)

    controller = Controller(dispatch_method)
    sys.modules["fastchat.serve.controller"].controller = controller

    # 解决默认的文档地址使用了国外CDN的问题，暂时不处理
    # MakeFastAPIOffline(app)
    app.title = "FastChat Controller"
    app._controller = controller
    return app


def run_controller(log_level: str = "INFO", started_event: mp.Event = None):
    import uvicorn
    import sys
    from server.utils import set_httpx_config

    set_httpx_config()

    app = create_controller_app(
        dispatch_method=FSCHAT_CONTROLLER.get("dispatch_method"),
        log_level=log_level,
    )
    set_app_event(app, started_event)

    host = FSCHAT_CONTROLLER["host"]
    port = FSCHAT_CONTROLLER["port"]

    if log_level == "ERROR":
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())
