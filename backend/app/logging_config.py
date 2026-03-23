"""
로깅 설정 모듈

표준 출력(터미널)에 [시간], [로그레벨], [메시지] 포맷으로 로그를 출력합니다.
"""

import logging
import sys
from datetime import datetime


def setup_logging(level: int = logging.INFO) -> None:
    """
    애플리케이션 로깅 설정
    
    Args:
        level: 로그 레벨 (기본값: INFO)
    """
    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 기존 핸들러 제거 (중복 방지)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 스트림 핸들러 (표준 출력)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    
    # 포매터: [시간], [로그레벨], [메시지]
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler.setFormatter(formatter)
    
    # 핸들러 추가
    root_logger.addHandler(stream_handler)
    
    # LangChain 및 기타 라이브러리의 verbose 로깅 제어
    logging.getLogger("langchain").setLevel(logging.INFO)
    logging.getLogger("langchain_core").setLevel(logging.INFO)
    logging.getLogger("langchain_google_genai").setLevel(logging.INFO)


# 애플리케이션 시작 시 자동으로 로깅 설정
setup_logging(logging.INFO)
