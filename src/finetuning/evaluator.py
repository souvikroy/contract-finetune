"""Model evaluation for fine-tuned models."""
import torch
from typing import List, Dict, Any, Optional
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import json

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class ModelEvaluator:
    """Evaluate fine-tuned model performance."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: str = "cuda"):
        """Initialize evaluator."""
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction()
    
    def evaluate_perplexity(self, dataloader: DataLoader) -> float:
        """Calculate perplexity on dataset."""
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item() * input_ids.size(0)
                total_tokens += input_ids.size(0)
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity
    
    def evaluate_bleu(self, predictions: List[str], references: List[str]) -> float:
        """Calculate BLEU score."""
        scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = nltk.word_tokenize(pred.lower())
            ref_tokens = nltk.word_tokenize(ref.lower())
            
            score = sentence_bleu(
                [ref_tokens],
                pred_tokens,
                smoothing_function=self.smoothing.method1
            )
            scores.append(score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def evaluate_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
            'rouge2': sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
            'rougeL': sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0
        }
    
    def evaluate_exact_match(self, predictions: List[str], references: List[str]) -> float:
        """Calculate exact match accuracy."""
        matches = sum(1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower())
        return matches / len(predictions) if predictions else 0.0
    
    def generate_predictions(self, examples: List[Dict[str, Any]], max_length: int = 512) -> List[str]:
        """Generate predictions for examples."""
        predictions = []
        
        with torch.no_grad():
            for example in examples:
                instruction = example.get('instruction', '')
                input_text = example.get('input', '')
                
                prompt = f"{instruction}\n\n{input_text}\n\nAnswer:"
                
                # Tokenize
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Generate
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract answer (after "Answer:")
                if "Answer:" in generated_text:
                    answer = generated_text.split("Answer:")[-1].strip()
                else:
                    answer = generated_text[len(prompt):].strip()
                
                predictions.append(answer)
        
        return predictions
    
    def evaluate_legal_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate legal domain-specific metrics."""
        # Citation accuracy (check for section/article references)
        citation_matches = 0
        for pred, ref in zip(predictions, references):
            # Extract citations (e.g., "Article 5.2", "Section 3.1")
            import re
            pred_citations = set(re.findall(r'(Article|Section|Clause)\s+[\d.]+', pred, re.IGNORECASE))
            ref_citations = set(re.findall(r'(Article|Section|Clause)\s+[\d.]+', ref, re.IGNORECASE))
            
            if pred_citations and ref_citations:
                if pred_citations.intersection(ref_citations):
                    citation_matches += 1
        
        citation_accuracy = citation_matches / len(predictions) if predictions else 0.0
        
        # Legal terminology check (simplified)
        legal_terms = ['contractor', 'obligation', 'liability', 'termination', 'compensation', 
                      'guarantee', 'defect', 'payment', 'clause', 'article', 'section']
        
        terminology_scores = []
        for pred in predictions:
            pred_lower = pred.lower()
            term_count = sum(1 for term in legal_terms if term in pred_lower)
            terminology_scores.append(min(term_count / len(legal_terms), 1.0))
        
        avg_terminology = sum(terminology_scores) / len(terminology_scores) if terminology_scores else 0.0
        
        return {
            'citation_accuracy': citation_accuracy,
            'legal_terminology_score': avg_terminology
        }
    
    def comprehensive_evaluate(
        self,
        test_examples: List[Dict[str, Any]],
        references: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Comprehensive evaluation."""
        # Generate predictions
        predictions = self.generate_predictions(test_examples)
        
        # Get references
        if references is None:
            references = [ex.get('output', '') for ex in test_examples]
        
        # Calculate metrics
        metrics = {
            'bleu': self.evaluate_bleu(predictions, references),
            'rouge': self.evaluate_rouge(predictions, references),
            'exact_match': self.evaluate_exact_match(predictions, references),
            'legal_metrics': self.evaluate_legal_metrics(predictions, references)
        }
        
        # Add sample predictions
        metrics['samples'] = [
            {
                'input': ex.get('input', '')[:200],
                'prediction': pred[:200],
                'reference': ref[:200]
            }
            for ex, pred, ref in zip(test_examples[:5], predictions[:5], references[:5])
        ]
        
        return metrics
    
    def save_evaluation_report(self, metrics: Dict[str, Any], output_path: str):
        """Save evaluation report to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
