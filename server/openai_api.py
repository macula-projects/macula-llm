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

# @File : openai_api.py
# @author: rain
# @Date：2023/11/18

import sys
from typing import List

from fastapi import FastAPI

from configs import (
    LOG_PATH,
    FSCHAT_OPENAI_API,
)
from server.utils import set_app_event, fschat_controller_address


def create_openai_api_app(
        controller_address: str,
        api_keys: List = [],
        log_level: str = "INFO",
) -> FastAPI:
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH
    from fastchat.serve.openai_api_server import app, CORSMiddleware, app_settings
    from fastchat.utils import build_logger
    logger = build_logger("openai_api", "openai_api.log")
    logger.setLevel(log_level)

    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    sys.modules["fastchat.serve.openai_api_server"].logger = logger
    app_settings.controller_address = controller_address
    app_settings.api_keys = api_keys

    # MakeFastAPIOffline(app)
    app.title = "FastChat OpeanAI API Server"
    return app


def run_openai_api(log_level: str = "INFO", started_event: mp.Event = None):
    import uvicorn
    import sys
    from server.utils import set_httpx_config

    set_httpx_config()

    controller_addr = fschat_controller_address()
    app = create_openai_api_app(controller_addr, log_level=log_level)  # TODO: not support keys yet.
    set_app_event(app, started_event)

    host = FSCHAT_OPENAI_API["host"]
    port = FSCHAT_OPENAI_API["port"]
    if log_level == "ERROR":
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    uvicorn.run(app, host=host, port=port)
