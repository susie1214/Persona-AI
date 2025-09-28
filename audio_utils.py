import pyaudio


def list_devices():
    pa = pyaudio.PyAudio()
    infos = []
    for i in range(pa.get_device_count()):
        di = pa.get_device_info_by_index(i)
        infos.append(di)
        print(
            f"[{i}] {di.get('name')} | hostAPI={di.get('hostApi')} | "
            f"maxInput={di.get('maxInputChannels')} | maxOutput={di.get('maxOutputChannels')} | "
            f"defaultSR={int(di.get('defaultSampleRate'))}"
        )
    pa.terminate()
    return infos


def pick_default_input_index():
    pa = pyaudio.PyAudio()
    try:
        idx = pa.get_default_input_device_info().get("index")
        return idx
    except Exception:
        return None
    finally:
        pa.terminate()


def open_input_stream(
    pa: pyaudio.PyAudio,
    rate=16000,
    channels=1,
    frames_per_buffer=1024,
    device_index=None,
):
    # 장치가 None이면 기본 입력 사용
    if device_index is None:
        try:
            device_index = pa.get_default_input_device_info().get("index")
        except Exception:
            raise RuntimeError(
                "기본 입력 장치를 찾지 못했습니다. 소리 설정에서 마이크 기본 장치를 지정하세요."
            )

    di = pa.get_device_info_by_index(device_index)
    if di.get("maxInputChannels", 0) < channels:
        raise RuntimeError(
            f"선택한 장치가 {channels} 채널 입력을 지원하지 않습니다: {di.get('name')}"
        )

    # 일부 장치는 16000을 지원하지 않습니다. 기본 SR로 자동 폴백
    sr = rate
    if int(di.get("defaultSampleRate")) not in (rate,):
        sr = int(di.get("defaultSampleRate"))  # 지원되는 기본값 사용

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=sr,
        input=True,
        output=False,  # 명확히 output 비활성화
        frames_per_buffer=frames_per_buffer,
        input_device_index=device_index,
    )
    return stream, sr
