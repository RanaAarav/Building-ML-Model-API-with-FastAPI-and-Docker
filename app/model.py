from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, time

class SentimentModel:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str):
        start = time.time()

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[0]
        label = ["negative", "positive"][torch.argmax(probs).item()]

        return {
            "sentiment": label,
            "confidence": probs.max().item(),
            "processing_time_ms": round((time.time() - start) * 1000, 3)
        }
