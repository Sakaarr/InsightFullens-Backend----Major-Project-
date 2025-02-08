from celery import shared_task
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

model_path = "C:/Users/DELL/Desktop/InsightFullens/insightfullens/product/t5base"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)
model.eval()

@shared_task
def extract_aspect_sentiment_task(reviews):
    """Celery task to run aspect-sentiment analysis asynchronously."""
    predictions = []
    for review in reviews:
        input_text = f"aspect-sentiment analysis: {review}"
        input_encoding = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_encoding["input_ids"],
                attention_mask=input_encoding["attention_mask"],
                max_length=50,
                num_beams=5,
                do_sample=True,
                temperature=0.7,
                top_k=50
            )

        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(prediction)

    return predictions
