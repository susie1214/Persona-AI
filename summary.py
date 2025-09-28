# summary.py
from typing import Dict, List
from openai import OpenAI
from config import Config

class SummaryService:
    def __init__(self, config: Config):
        self.config = config
        self.client = OpenAI(api_key=config.api_key) if config.api_key else None
        
    def is_available(self) -> bool:
        """요약 서비스 사용 가능 여부 확인"""
        return self.client is not None
    
    def summarize_speaker_text(self, speaker: str, texts: List[str]) -> str:
        """특정 화자의 텍스트를 요약"""
        if not self.is_available():
            return "(OpenAI API 키가 설정되지 않아 요약을 사용할 수 없습니다)"
        
        if not texts:
            return "(전사된 내용이 없습니다)"
        
        combined_text = " ".join(texts)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "한국어 텍스트를 간결하고 명확하게 요약해주세요."},
                    {"role": "user", "content": combined_text}
                ],
                max_tokens=1024,
                temperature=0.5
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"(요약 실패: {e})"
    
    def summarize_all_speakers(self, speaker_texts: Dict[str, List[str]]) -> Dict[str, str]:
        """모든 화자의 텍스트를 요약"""
        summaries = {}
        for speaker, texts in speaker_texts.items():
            summaries[speaker] = self.summarize_speaker_text(speaker, texts)
        return summaries
    
    def create_meeting_summary(self, speaker_texts: Dict[str, List[str]]) -> str:
        """전체 회의 요약 생성"""
        if not self.is_available():
            return "(OpenAI API 키가 설정되지 않아 요약을 사용할 수 없습니다)"
        
        if not speaker_texts:
            return "(전사된 내용이 없습니다)"
        
        # 모든 화자의 내용을 시간순으로 합치기 (실제로는 순서를 보장하기 위해 추가 구현 필요)
        all_text = []
        for speaker, texts in speaker_texts.items():
            for text in texts:
                all_text.append(f"{speaker}: {text}")
        
        combined_text = "\n".join(all_text)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system", 
                        "content": "다음은 회의 내용입니다. 주요 논점과 결론을 정리하여 회의록 형태로 요약해주세요."
                    },
                    {"role": "user", "content": combined_text}
                ],
                max_tokens=2048,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"(전체 요약 실패: {e})"
    
    def create_custom_summary(self, text: str, prompt: str) -> str:
        """사용자 정의 프롬프트로 요약"""
        if not self.is_available():
            return "(OpenAI API 키가 설정되지 않아 요약을 사용할 수 없습니다)"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": text}
                ],
                max_tokens=1024,
                temperature=0.5
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"(요약 실패: {e})"