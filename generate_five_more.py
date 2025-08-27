#!/usr/bin/env python3
"""
Generate Five More Data Entries Using Ollama
This script uses Ollama to generate 5 additional high-quality FAQ entries
based on your current dataset and specific AI model deployment scenarios.
"""

import json
import subprocess
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FocusedOllamaGenerator:
    def __init__(self):
        self.ollama_model = "llama3.2:1b"  # Using a smaller model for faster generation
        self.new_entries = []
        
    def call_ollama(self, prompt: str) -> str:
        """Call Ollama to generate a response"""
        try:
            # Create a focused prompt for better responses
            enhanced_prompt = f"""You are an AI model deployment expert. Please provide a concise, technical answer to this question. Focus on practical deployment advice and real-world considerations.

Question: {prompt}

Answer:"""
            
            result = subprocess.run([
                "ollama", "run", self.ollama_model, enhanced_prompt
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.warning(f"Ollama call failed: {result.stderr}")
                return ""
                
        except Exception as e:
            logger.warning(f"Ollama call failed: {e}")
            return ""
    
    def generate_five_entries(self):
        """Generate exactly 5 new high-quality FAQ entries"""
        logger.info("🚀 Generating 5 new FAQ entries using Ollama...")
        
        # Define 5 specific, high-value questions for AI model deployment
        questions = [
            {
                "question": "How do I optimize memory usage when deploying large language models on limited GPU resources?",
                "category": "optimization",
                "difficulty": "advanced",
                "focus": "memory_optimization"
            },
            {
                "question": "What are the best practices for implementing A/B testing with different AI model variants in production?",
                "category": "deployment",
                "difficulty": "intermediate",
                "focus": "production_testing"
            },
            {
                "question": "How can I implement automatic model versioning and rollback mechanisms for AI model deployments?",
                "category": "deployment",
                "difficulty": "advanced",
                "focus": "version_management"
            },
            {
                "question": "What are the security considerations when deploying AI models that handle sensitive user data?",
                "category": "security",
                "difficulty": "intermediate",
                "focus": "data_security"
            },
            {
                "question": "How do I implement real-time monitoring and alerting for AI model performance in production environments?",
                "category": "monitoring",
                "difficulty": "intermediate",
                "focus": "production_monitoring"
            }
        ]
        
        for i, q_info in enumerate(questions, 1):
            logger.info(f"🔍 Generating entry {i}/5: {q_info['question'][:60]}...")
            
            # Use Ollama to generate the answer
            answer = self.call_ollama(q_info["question"])
            
            if answer:
                # Clean up the answer (remove extra whitespace and newlines)
                answer = " ".join(answer.split()).strip()
                
                entry = {
                    "id": len(self.new_entries) + 16,  # Continue from where comprehensive data left off
                    "question": q_info["question"],
                    "answer": answer,
                    "model_name": "AI Model Deployment",
                    "category": q_info["category"],
                    "difficulty": q_info["difficulty"],
                    "source": "ollama_generated",
                    "focus_area": q_info["focus"],
                    "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                self.new_entries.append(entry)
                logger.info(f"✅ Generated entry {i}/5 successfully")
                
                # Small delay to avoid overwhelming Ollama
                time.sleep(1)
            else:
                logger.warning(f"⚠️ Failed to generate answer for entry {i}")
        
        logger.info(f"🎉 Successfully generated {len(self.new_entries)} new entries")
        return self.new_entries
    
    def save_enhanced_dataset(self):
        """Save the enhanced dataset with the new entries"""
        logger.info("💾 Saving enhanced dataset...")
        
        # Load existing comprehensive data
        existing_file = "data/comprehensive_faq_docs.jsonl"
        if Path(existing_file).exists():
            with open(existing_file, 'r', encoding='utf-8') as f:
                existing_data = [json.loads(line) for line in f]
            
            # Combine existing and new data
            enhanced_data = existing_data + self.new_entries
            
            # Save enhanced dataset
            with open("data/enhanced_faq_docs.jsonl", 'w', encoding='utf-8') as f:
                for entry in enhanced_data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            # Update metadata
            self._update_metadata(enhanced_data)
            
            logger.info(f"✅ Enhanced dataset saved with {len(enhanced_data)} total entries")
            return True
        else:
            logger.error(f"❌ Existing comprehensive data file not found: {existing_file}")
            return False
    
    def _update_metadata(self, enhanced_data):
        """Update the dataset metadata"""
        metadata = {
            "dataset_info": {
                "name": "Enhanced AI Model FAQ Dataset (Ollama Enhanced)",
                "version": "4.0",
                "description": "Comprehensive dataset with Ollama-generated additional entries",
                "total_entries": len(enhanced_data),
                "source": "github_issues_analysis + ollama_generation",
                "ollama_model_used": self.ollama_model,
                "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "coverage": {
                "deployment_questions": len([e for e in enhanced_data if e["category"] == "deployment"]),
                "optimization_questions": len([e for e in enhanced_data if e["category"] == "optimization"]),
                "integration_questions": len([e for e in enhanced_data if e["category"] == "integration"]),
                "evaluation_questions": len([e for e in enhanced_data if e["category"] == "evaluation"]),
                "security_questions": len([e for e in enhanced_data if e["category"] == "security"]),
                "maintenance_questions": len([e for e in enhanced_data if e["category"] == "maintenance"]),
                "monitoring_questions": len([e for e in enhanced_data if e["category"] == "monitoring"])
            },
            "ollama_enhancement": {
                "new_entries_generated": len(self.new_entries),
                "focus_areas": list(set([e.get("focus_area", "") for e in self.new_entries if e.get("focus_area")])),
                "categories_added": list(set([e["category"] for e in self.new_entries]))
            }
        }
        
        # Save updated metadata
        with open("data/enhanced_dataset_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ Dataset metadata updated")
    
    def run_generation(self):
        """Run the complete generation process"""
        logger.info("🚀 Starting focused Ollama data generation...")
        
        # Check if Ollama is available
        try:
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ Ollama available: {result.stdout.strip()}")
            else:
                logger.error("❌ Ollama not available")
                return False
        except FileNotFoundError:
            logger.error("❌ Ollama not found")
            return False
        
        # Generate the 5 new entries
        new_entries = self.generate_five_entries()
        
        if new_entries:
            # Save the enhanced dataset
            if self.save_enhanced_dataset():
                logger.info("🎉 Data generation completed successfully!")
                return True
        
        return False

def main():
    """Main execution function"""
    generator = FocusedOllamaGenerator()
    
    try:
        success = generator.run_generation()
        
        if success:
            print("\n🎉 SUCCESS! Generated 5 new FAQ entries using Ollama!")
            print(f"\n📊 New entries created:")
            for i, entry in enumerate(generator.new_entries, 1):
                print(f"  {i}. {entry['question'][:80]}...")
                print(f"     Category: {entry['category']}, Difficulty: {entry['difficulty']}")
                print(f"     Focus: {entry.get('focus_area', 'N/A')}")
                print()
            
            print("📁 Enhanced dataset saved to: data/enhanced_faq_docs.jsonl")
            print("📋 Updated metadata: data/enhanced_dataset_metadata.json")
            print("\n🚀 Next steps:")
            print("1. Run: python scripts/preprocess.py (to update indexes)")
            print("2. Run: python eval_easy.py (to test with new data)")
            print("3. Check artifacts/ for updated results")
            
        else:
            print("\n❌ Data generation failed. Check logs for details.")
            
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        print(f"\n⚠️ Unexpected error: {e}")

if __name__ == "__main__":
    main()
