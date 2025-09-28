from core.llm_openai import OpenAILLM
from core.llm_ax import AXLLM
from core.llm_midm import MidmLLM

def run_all_models(prompt="안녕하세요, 자기소개 부탁드립니다."):
    results = {}

    try:
        openai_llm = OpenAILLM("gpt-4o-mini")
        results["OpenAI"] = openai_llm.complete(prompt, temperature=0.3)
    except Exception as e:
        results["OpenAI"] = f"❌ Error: {e}"

    try:
        ax_llm = AXLLM("skt/A.X-4.0")
        results["A.X"] = ax_llm.complete(prompt, temperature=0.7)
    except Exception as e:
        results["A.X"] = f"❌ Error: {e}"

    try:
        midm_llm = MidmLLM("K-intelligence/Midm-2.0-Mini-Instruct")
        results["Midm-2.0"] = midm_llm.complete(prompt, temperature=0.7)
    except Exception as e:
        results["Midm-2.0"] = f"❌ Error: {e}"

    return results

if __name__ == "__main__":
    prompt = "대한민국의 인공지능 산업 현황을 간단히 요약해줘."
    outputs = run_all_models(prompt)
    for k, v in outputs.items():
        print("="*60)
        print(f"💡 {k} 결과:\n{v}\n")
