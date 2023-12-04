import json
from threading import Lock
from typing import (
    Optional,
    Union,
    Iterator,
    Dict,
    Any,
    AsyncIterator,
)

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool

from api.config import SETTINGS
from api.utils.constants import ErrorCode
from api.utils.protocol import (
    ChatCompletionCreateParams,
    CompletionCreateParams,
    ErrorResponse,
)

llama_outer_lock = Lock()
llama_inner_lock = Lock()


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
):
    if SETTINGS.api_keys:
        if auth is None or (token := auth.credentials) not in SETTINGS.api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token
    else:
        # api_keys not set; allow all
        return None


def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(ErrorResponse(message=message, code=code).model_dump(), status_code=500)


async def handle_request(
    request: Union[CompletionCreateParams, ChatCompletionCreateParams],
    stop: Dict[str, Any] = None,
    chat: bool = True,
):
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret

    # stop settings
    _stop, stop_token_ids = [], []
    if stop is not None:
        stop_token_ids = stop.get("token_ids", [])
        _stop = stop.get("strings", [])

    request.stop = request.stop or []
    if isinstance(request.stop, str):
        request.stop = [request.stop]

    if chat:
        if "qwen" in SETTINGS.model_name.lower() and request.functions:
            request.stop.append("Observation:")

    request.stop = list(set(_stop + request.stop))

    return request, stop_token_ids


def check_requests(request: Union[CompletionCreateParams, ChatCompletionCreateParams]) -> Optional[JSONResponse]:
    # Check all params
    if request.max_tokens is not None and request.max_tokens <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.max_tokens} is less than the minimum of 1 - 'max_tokens'",
        )
    if request.n is not None and request.n <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.n} is less than the minimum of 1 - 'n'",
        )
    if request.temperature is not None and request.temperature < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is less than the minimum of 0 - 'temperature'",
        )
    if request.temperature is not None and request.temperature > 2:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is greater than the maximum of 2 - 'temperature'",
        )
    if request.top_p is not None and request.top_p < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is less than the minimum of 0 - 'top_p'",
        )
    if request.top_p is not None and request.top_p > 1:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is greater than the maximum of 1 - 'temperature'",
        )
    if request.stop is not None and (
            not isinstance(request.stop, str) and not isinstance(request.stop, list)
    ):
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.stop} is not valid under any of the given schemas - 'stop'",
        )
    return None


async def get_event_publisher(
    request: Request,
    inner_send_chan: MemoryObjectSendStream,
    iterator: Union[Iterator, AsyncIterator],
):
    async with inner_send_chan:
        try:
            if SETTINGS.engine != "vllm":
                async for chunk in iterate_in_threadpool(iterator):
                    if isinstance(chunk, BaseModel):
                        chunk = chunk.model_dump_json()
                    elif isinstance(chunk, dict):
                        chunk = json.dumps(chunk, ensure_ascii=False)

                    await inner_send_chan.send(dict(data=chunk))

                    if await request.is_disconnected():
                        raise anyio.get_cancelled_exc_class()()

                    if SETTINGS.interrupt_requests and llama_outer_lock.locked():
                        await inner_send_chan.send(dict(data="[DONE]"))
                        raise anyio.get_cancelled_exc_class()()
            else:
                async for chunk in iterator:
                    chunk = chunk.model_dump_json()
                    await inner_send_chan.send(dict(data=chunk))
                    if await request.is_disconnected():
                        raise anyio.get_cancelled_exc_class()()
            await inner_send_chan.send(dict(data="[DONE]"))
        except anyio.get_cancelled_exc_class() as e:
            logger.info("disconnected")
            with anyio.move_on_after(1, shield=True):
                logger.info(f"Disconnected from client (via refresh/close) {request.client}")
                raise e
