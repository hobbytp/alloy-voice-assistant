# coding=utf-8
#
# Installation instructions for pyaudio:
# APPLE Mac OS X
#   brew install portaudio
#   pip install pyaudio
# Debian/Ubuntu
#   sudo apt-get install python-pyaudio python3-pyaudio
#   or
#   pip install pyaudio
# CentOS
#   sudo yum install -y portaudio portaudio-devel && pip install pyaudio
# Microsoft Windows
#   python -m pip install pyaudio

import unittest
import time
import pyaudio
import dashscope
from dashscope.api_entities.dashscope_response import SpeechSynthesisResponse
from dashscope.audio.tts_v2 import *


class CosyVoiceCallback(ResultCallback):
    _player = None
    _stream = None

    def on_open(self):
        print("websocket is open.")
        self._player = pyaudio.PyAudio()
        self._stream = self._player.open(
            format=pyaudio.paInt16, channels=1, rate=22050, output=True
        )

    def on_complete(self):
        print("speech synthesis task complete successfully.")

    def on_error(self, message: str):
        print(f"speech synthesis task failed, {message}")

    def on_close(self):
        print("websocket is closed.")
        # 停止播放器
        self._stream.stop_stream()
        self._stream.close()
        self._player.terminate()

    def on_event(self, message):
        print(f"recv speech synthsis message {message}")

    def on_data(self, data: bytes) -> None:
        print("audio result length:", len(data))
        self._stream.write(data)


class TestExample(unittest.TestCase):
    dashscope.api_key = "your-dashscope-api-key"
    model = "cosyvoice-v1"
    voice = "longxiaochun"

    def test_example(self):
        callback = CosyVoiceCallback()

        test_text = [
            "流式文本语音合成SDK，",
            "可以将输入的文本",
            "合成为语音二进制数据，",
            "相比于非流式语音合成，",
            "流式合成的优势在于实时性",
            "更强。用户在输入文本的同时",
            "可以听到接近同步的语音输出，",
            "极大地提升了交互体验，",
            "减少了用户等待时间。",
            "适用于调用大规模",
            "语言模型（LLM），以",
            "流式输入文本的方式",
            "进行语音合成的场景。",
        ]

        synthesizer = SpeechSynthesizer(
            model=self.model,
            voice=self.voice,
            format=AudioFormat.PCM_22050HZ_MONO_16BIT,
            callback=callback,
        )

        for text in test_text:
            synthesizer.streaming_call(text)
            time.sleep(0.5)
        synthesizer.streaming_complete()
        print("requestId: ", synthesizer.get_last_request_id())

        # Assuming there's a function named 'example_function' in the code
        # and we want to test it
        # self.assertEqual(example_function(1, 2), 3,"example_function(1, 2) should return 3")


if __name__ == "__main__":
    unittest.main()
