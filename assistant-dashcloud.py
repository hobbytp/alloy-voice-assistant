import base64
import os
from threading import Lock, Thread

import cv2
import openai
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError
from speech_recognition import AudioData

import pyttsx3
import dashscope
from dashscope.audio.tts_v2 import *
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess




import argparse


load_dotenv()


class WebcamStream:
    def __init__(self):
        self.stream = VideoCapture(index=0)
        _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self

        self.running = True

        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.running:
            _, frame = self.stream.read()

            self.lock.acquire()
            self.frame = frame
            self.lock.release()

    def read(self, encode=False):
        self.lock.acquire()
        frame = self.frame.copy()
        self.lock.release()

        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()


class Assistant:
    def __init__(self, model, tts_engine):
        self.chain = self._create_inference_chain(model)
        if tts_engine == "dashscope":
            self._tts = self._tts_dashscope
        elif tts_engine == "pyttsx3":
            self._tts = self._tts_pyttsx3
            self.engine = pyttsx3.init()
        elif tts_engine == "openai":
            self._tts = self._tts_openai

    def answer(self, prompt, image):
        if not prompt:
            return

        print("Prompt:", prompt)

        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)

    def _tts_dashscope(self, response):
        '''
        Call Ali Qwen Cosyvoice API to synthesize speech.
        '''
        dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]
        model = "cosyvoice-v1"
        voice = "longxiaochun"

        synthesizer = SpeechSynthesizer(model=model, voice=voice)
        audio = synthesizer.call(response)
        print('requestId: ', synthesizer.get_last_request_id())
        # with open('output.mp3', 'wb') as f:
        #    f.write(audio)

    def _tts_pyttsx3(self, response):
        self.engine.say(response)
        self.engine.runAndWait()
        
        
    def _tts_openai(self, response):    
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",
            input=response,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)
        

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions.

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. Do not ask the user any questions.

        Be friendly and helpful. Show some personality. Do not be too formal.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )

    def audio_callback(self, recognizer, audio):
        '''
        whisper model will be auto download from https://openaipublic.azureedge.net/main/whisper/models/
        中文将由多语言支持的model提供支持，所以对中文的语音识别的支持没有国内的语言识别系统好，比如粤语，方言等。
        '''
        try:
            
            # prompt = recognizer.recognize_whisper(
            #    audio, model="base", language="english")
            # prompt = recognizer.recognize_whisper(
            #    audio, model="base", language="chinese")  #see more languages: https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
            if isinstance(recognizer, QwenRecognizer) :
                prompt = recognizer.recognize_funasr(
                    audio, model="base", language="chinese")
            elif isinstance(recognizer, Recognizer) :
                prompt = recognizer.recognize_whisper(
                    audio, model="base", language="english")
                
            assistant.answer(prompt, webcam_stream.read(encode=True))
            

        except UnknownValueError:
            print("There was an error processing the audio.")

class QwenRecognizer(Recognizer):

    def recognize_funasr(self, audio_data, model="base", show_dict=False, load_options=None, language=None, translate=False, **transcribe_options):
        """
        Performs speech recognition on ``audio_data`` (an ``AudioData`` instance), using Whisper.

        The recognition language is determined by ``language``, an uncapitalized full language name like "english" or "chinese". See the full language list at https://github.com/openai/whisper/blob/main/whisper/tokenizer.py

        model can be any of tiny, base, small, medium, large, tiny.en, base.en, small.en, medium.en. See https://github.com/openai/whisper for more details.

        If show_dict is true, returns the full dict response from Whisper, including the detected language. Otherwise returns only the transcription.

        You can translate the result to english with Whisper by passing translate=True

        Other values are passed directly to whisper. See https://github.com/openai/whisper/blob/main/whisper/transcribe.py for all options
        """

        assert isinstance(audio_data, AudioData), "Data must be audio data"
        import numpy as np
        import soundfile as sf
        import torch
        import whisper
        import io
        '''
        if load_options or not hasattr(self, "whisper_model") or self.whisper_model.get(model) is None:
            self.whisper_model = getattr(self, "whisper_model", {})
            self.whisper_model[model] = whisper.load_model(model, **load_options or {})

        # 16 kHz https://github.com/openai/whisper/blob/28769fcfe50755a817ab922a7bc83483159600a9/whisper/audio.py#L98-L99
        wav_bytes = audio_data.get_wav_data(convert_rate=16000)
        wav_stream = io.BytesIO(wav_bytes)
        audio_array, sampling_rate = sf.read(wav_stream)
        audio_array = audio_array.astype(np.float32)

        result = self.whisper_model[model].transcribe(
            audio_array,
            language=language,
            task="translate" if translate else None,
            fp16=torch.cuda.is_available(),
            **transcribe_options
        )        
        if show_dict:
            return result
        else:
            return result["text"]
        '''
        # 16 kHz https://github.com/openai/whisper/blob/28769fcfe50755a817ab922a7bc83483159600a9/whisper/audio.py#L98-L99
        wav_bytes = audio_data.get_wav_data(convert_rate=16000)
        wav_stream = io.BytesIO(wav_bytes)
        audio_array, sampling_rate = sf.read(wav_stream)
        audio_array = audio_array.astype(np.float32)

        model_dir = "iic/SenseVoiceSmall"
        model = AutoModel(
            model=model_dir,
            trust_remote_code=True,
            remote_code="./model.py",  
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
        )
        res = model.generate(
            #input=f"{model.model_path}/example/en.mp3", 
            input= wav_bytes, # model.inference 支持input是byte， 参考prepare_data_iterator函数
            cache={},
            language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=15,
        )
        text = rich_transcription_postprocess(res[0]["text"])
        print(text)
        return text
    

# 大模型需要时多模态的，因为有图片输入。目前有Google的Gemini和Qwen-LV等， mistral不能支持。
# 语音要能识别中文和英文，并能TTS输出。语音识别可以是python的包，TTS需要大模型或本地的包。
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assistant Configuration")
    parser.add_argument("-p", "--model_provider", type=str,
                        default="google", help="LLM provider")
    parser.add_argument("-m", "--model", type=str,
                        default="gemini-1.5-flash-latest", help="Inference model name")
    parser.add_argument("-t", "--tts_engine", type=str,
                        default="dashscope", help="TTS engine provider") #dashscope, pyttsx3, openai
    parser.add_argument("-r", "--recognizer", type=str,
                        default="base", help="Speech recognizer provider")
    args = parser.parse_args()

    model_provider = args.model_provider
    model_name = args.model
    tts_engine = args.tts_engine
    recognizer_provider = args.recognizer

    webcam_stream = WebcamStream().start()
    if model_provider == "google" and model_name == "gemini-1.5-flash-latest":
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    elif model_provider == "groq":
        model = ChatGroq(model_name=model_name,
                         groq_api_key=os.environ.get("GROQ_API_KEY"))

    assistant = Assistant(model_name, tts_engine=tts_engine)

    # recognizer voice from microphone.
    if recognizer_provider == "base":
        recognizer = Recognizer()  # speech_recognizer 内部支持google_cloud, azure, ibm, whisper 等。
    elif recognizer_provider == "qwen":
        recognizer = QwenRecognizer()
    #elif recognizer_provider = "baidu":
    #    recognizer = BaiduRecognizer()
    else:
        raise ValueError("Invalid recognizer provider")

    microphone = Microphone()
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)

    stop_listening = recognizer.listen_in_background(
        microphone, assistant.audio_callback)  #source is microphone, and then send audiodata to callback func.

    while True:
        cv2.imshow("webcam", webcam_stream.read())
        if cv2.waitKey(1) in [27, ord("q")]:
            break

    webcam_stream.stop()
    cv2.destroyAllWindows()
    stop_listening(wait_for_stop=False)
