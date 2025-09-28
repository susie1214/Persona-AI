# core/adapter.py
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    PEFT_AVAILABLE = True
except Exception:
    PEFT_AVAILABLE = False


class AdapterManager:
    def __init__(self):
        self.available = PEFT_AVAILABLE
        self.base_model_id = None
        self.base_model = None
        self.tokenizer = None
        self.loaded_adapters = {}
        self.active_adapter = None

    def load_base(self, base_model_id="EleutherAI/pythia-410m"):
        if not self.available:
            return False
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            self.base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
            self.base_model_id = base_model_id
            return True
        except Exception as e:
            print("[WARN] load_base failed:", e)
            return False

    def load_adapter(self, name: str, adapter_path: str):
        if not self.available or self.base_model is None:
            return False
        try:
            m = PeftModel.from_pretrained(self.base_model, adapter_path)
            self.loaded_adapters[name] = m
            return True
        except Exception as e:
            print("[WARN] load_adapter failed:", e)
            return False

    def set_active(self, name):
        self.active_adapter = name if name in self.loaded_adapters else None

    def respond(self, prompt: str) -> str:
        if self.active_adapter:
            return f"(어댑터:{self.active_adapter}) {prompt} -> 친근하고 공손한 톤으로 답변합니다."
        return f"{prompt} -> 기본 톤 답변(데모)."
