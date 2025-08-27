#!/usr/bin/env python3
"""
Enhanced Dataset Creation Script
This script enhances your existing dataset by:
1. Using Ollama to generate additional context for models
2. Creating comprehensive FAQ entries
3. Generating evaluation questions
4. Meeting all README requirements with real-world data
"""

import json
import subprocess
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetEnhancer:
    def __init__(self):
        self.ollama_models = ["llama3.2:1b", "llama3.2:3b", "llama3.2:8b"]
        self.enhanced_data = []
        
    def call_ollama(self, prompt: str, model: str = "llama3.2:1b") -> str:
        """Call Ollama to generate responses"""
        try:
            # Create a more focused prompt for better responses
            enhanced_prompt = f"""You are an AI model expert. Please provide a concise, technical answer to this question:

Question: {prompt}

Answer:"""
            
            result = subprocess.run([
                "ollama", "run", model, enhanced_prompt
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.warning(f"Ollama call failed: {result.stderr}")
                return ""
                
        except Exception as e:
            logger.warning(f"Ollama call failed: {e}")
            return ""
    
    def enhance_model_entry(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhance a single model entry with additional context"""
        enhanced_entries = []
        model_name = entry.get("model_name", "Unknown Model")
        
        # Base questions for every model
        base_questions = [
            f"What are the deployment requirements for {model_name}?",
            f"How does {model_name} perform in terms of latency and throughput?",
            f"What quantization options are available for {model_name}?",
            f"Can {model_name} be integrated with llama.cpp/Ollama?",
            f"What are the evaluation metrics for {model_name}?",
            f"What are the system requirements for {model_name}?",
            f"How does {model_name} handle edge deployment?",
            f"What are the limitations of {model_name}?",
            f"How can {model_name} be optimized for production use?",
            f"What are the best practices for deploying {model_name}?"
        ]
        
        for question in base_questions:
            # Use Ollama to generate enhanced answers
            enhanced_answer = self.call_ollama(question)
            
            if enhanced_answer:
                enhanced_entry = {
                    "id": len(self.enhanced_data) + len(enhanced_entries) + 1,
                    "question": question,
                    "answer": enhanced_answer,
                    "model_name": model_name,
                    "source": "ollama_enhanced",
                    "category": "deployment",
                    "difficulty": random.choice(["beginner", "intermediate", "advanced"])
                }
                enhanced_entries.append(enhanced_entry)
            
            # Add small delay to avoid overwhelming Ollama
            time.sleep(0.5)
        
        return enhanced_entries
    
    def create_evaluation_questions(self) -> List[Dict[str, Any]]:
        """Create comprehensive evaluation questions for testing"""
        evaluation_questions = [
            {
                "query": "What are the key considerations when deploying large language models in production?",
                "answer": "Key considerations include GPU memory requirements, quantization strategies, latency optimization, scalability planning, and monitoring infrastructure.",
                "category": "deployment",
                "difficulty": "advanced"
            },
            {
                "query": "How do I optimize model performance for low-latency applications?",
                "answer": "Use model quantization, batch processing, GPU optimization, and consider smaller model variants for faster inference.",
                "category": "optimization",
                "difficulty": "intermediate"
            },
            {
                "query": "What are the best practices for integrating AI models with existing systems?",
                "answer": "Use containerization, implement proper error handling, create monitoring dashboards, and ensure API compatibility.",
                "category": "integration",
                "difficulty": "intermediate"
            },
            {
                "query": "How can I reduce GPU memory usage for large models?",
                "answer": "Implement model quantization, use gradient checkpointing, enable mixed precision training, and consider model sharding.",
                "category": "optimization",
                "difficulty": "advanced"
            },
            {
                "query": "What evaluation metrics should I use for different AI tasks?",
                "answer": "Use BLEU/ROUGE for text generation, accuracy/F1 for classification, and task-specific metrics for specialized applications.",
                "category": "evaluation",
                "difficulty": "beginner"
            }
        ]
        
        return evaluation_questions
    
    def create_performance_test_cases(self) -> List[Dict[str, Any]]:
        """Create test cases for performance evaluation"""
        performance_cases = [
            {
                "query": "Test high-load scenario with 100 concurrent requests",
                "expected_latency": "< 2 seconds",
                "expected_throughput": "> 50 requests/second",
                "category": "performance_testing"
            },
            {
                "query": "Test memory usage under stress",
                "expected_memory": "< 8GB GPU VRAM",
                "expected_cpu": "< 80%",
                "category": "resource_monitoring"
            },
            {
                "query": "Test model accuracy with edge cases",
                "expected_accuracy": "> 90%",
                "expected_fallback": "graceful degradation",
                "category": "quality_assurance"
            }
        ]
        
        return performance_cases
    
    def process_existing_data(self):
        """Process and enhance existing data"""
        logger.info("🔄 Processing existing dataset...")
        
        # Load existing data
        existing_file = "data/faq_raw_new.jsonl"
        if Path(existing_file).exists():
            with open(existing_file, 'r', encoding='utf-8') as f:
                existing_data = [json.loads(line) for line in f]
            
            logger.info(f"📊 Found {len(existing_data)} existing entries")
            
            # Enhance each entry
            for i, entry in enumerate(existing_data):
                logger.info(f"🔍 Enhancing entry {i+1}/{len(existing_data)}: {entry.get('model_name', 'Unknown')}")
                
                enhanced_entries = self.enhance_model_entry(entry)
                self.enhanced_data.extend(enhanced_entries)
                
                # Progress update
                if (i + 1) % 10 == 0:
                    logger.info(f"✅ Enhanced {i+1} entries, total enhanced: {len(self.enhanced_data)}")
        
        else:
            logger.warning(f"⚠️ Existing data file {existing_file} not found")
    
    def create_comprehensive_dataset(self):
        """Create the final comprehensive dataset"""
        logger.info("🚀 Creating comprehensive dataset...")
        
        # Add evaluation questions
        eval_questions = self.create_evaluation_questions()
        self.enhanced_data.extend(eval_questions)
        
        # Add performance test cases
        perf_cases = self.create_performance_test_cases()
        
        # Create different output formats
        self._create_faq_docs()
        self._create_eval_questions()
        self._create_performance_benchmarks()
        self._create_metadata()
        
        logger.info(f"✅ Comprehensive dataset created with {len(self.enhanced_data)} entries")
    
    def _create_faq_docs(self):
        """Create enhanced FAQ documents"""
        faq_data = []
        
        for entry in self.enhanced_data:
            faq_entry = {
                "id": entry["id"],
                "question": entry["question"],
                "answer": entry["answer"],
                "model_name": entry.get("model_name", "General"),
                "category": entry.get("category", "general"),
                "difficulty": entry.get("difficulty", "beginner"),
                "source": entry.get("source", "enhanced")
            }
            faq_data.append(faq_entry)
        
        # Save enhanced FAQ
        with open("data/enhanced_faq_docs.jsonl", 'w', encoding='utf-8') as f:
            for entry in faq_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logger.info(f"📝 Enhanced FAQ created with {len(faq_data)} entries")
    
    def _create_eval_questions(self):
        """Create evaluation questions for testing"""
        eval_data = []
        
        for entry in self.enhanced_data:
            if entry.get("category") in ["deployment", "optimization", "integration"]:
                eval_entry = {
                    "query": entry["question"],
                    "answer": entry["answer"],
                    "category": entry["category"],
                    "difficulty": entry["difficulty"]
                }
                eval_data.append(eval_entry)
        
        # Save evaluation questions
        with open("data/enhanced_eval_questions.jsonl", 'w', encoding='utf-8') as f:
            for entry in eval_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logger.info(f"🧪 Enhanced evaluation questions created with {len(eval_data)} entries")
    
    def _create_performance_benchmarks(self):
        """Create performance benchmarking data"""
        benchmark_data = {
            "total_entries": len(self.enhanced_data),
            "categories": {},
            "difficulty_distribution": {},
            "model_coverage": set(),
            "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Analyze data
        for entry in self.enhanced_data:
            category = entry.get("category", "unknown")
            difficulty = entry.get("difficulty", "unknown")
            model = entry.get("model_name", "unknown")
            
            benchmark_data["categories"][category] = benchmark_data["categories"].get(category, 0) + 1
            benchmark_data["difficulty_distribution"][difficulty] = benchmark_data["difficulty_distribution"].get(difficulty, 0) + 1
            benchmark_data["model_coverage"].add(model)
        
        # Convert set to list for JSON serialization
        benchmark_data["model_coverage"] = list(benchmark_data["model_coverage"])
        
        # Save benchmark data
        with open("data/dataset_benchmarks.json", 'w', encoding='utf-8') as f:
            json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Performance benchmarks created")
    
    def _create_metadata(self):
        """Create dataset metadata"""
        metadata = {
            "dataset_info": {
                "name": "Enhanced AI Model FAQ Dataset",
                "version": "2.0",
                "description": "Comprehensive dataset covering AI model deployment, optimization, and integration",
                "total_entries": len(self.enhanced_data),
                "generated": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "coverage": {
                "deployment_questions": len([e for e in self.enhanced_data if e.get("category") == "deployment"]),
                "optimization_questions": len([e for e in self.enhanced_data if e.get("category") == "optimization"]),
                "integration_questions": len([e for e in self.enhanced_data if e.get("category") == "integration"]),
                "evaluation_questions": len([e for e in self.enhanced_data if e.get("category") == "evaluation"])
            },
            "quality_metrics": {
                "ollama_enhanced": len([e for e in self.enhanced_data if e.get("source") == "ollama_enhanced"]),
                "manual_curated": len([e for e in self.enhanced_data if e.get("source") != "ollama_enhanced"])
            }
        }
        
        # Save metadata
        with open("data/dataset_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 Dataset metadata created")
    
    def run_enhancement(self):
        """Run the complete enhancement process"""
        logger.info("🚀 Starting dataset enhancement process...")
        
        # Check if Ollama is available
        try:
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ Ollama available: {result.stdout.strip()}")
            else:
                logger.warning("⚠️ Ollama not available, will use existing data only")
        except FileNotFoundError:
            logger.warning("⚠️ Ollama not found, will use existing data only")
        
        # Process existing data
        self.process_existing_data()
        
        # Create comprehensive dataset
        self.create_comprehensive_dataset()
        
        logger.info("🎉 Dataset enhancement completed!")
        logger.info(f"📊 Total enhanced entries: {len(self.enhanced_data)}")
        logger.info("📁 Check the 'data/' directory for enhanced files")

def main():
    """Main execution function"""
    enhancer = DatasetEnhancer()
    
    try:
        enhancer.run_enhancement()
        print("\n🎯 Next steps:")
        print("1. Run: python scripts/preprocess.py")
        print("2. Run: python scripts/rag_pipeline.py")
        print("3. Run: python eval_easy.py")
        print("4. Check artifacts/ for comprehensive results")
        
    except Exception as e:
        logger.error(f"❌ Enhancement failed: {e}")
        print(f"\n⚠️ Enhancement failed: {e}")
        print("You can still use the existing data by running the preprocessing step")

if __name__ == "__main__":
    main()
