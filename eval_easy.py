import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging
from scripts.rag_pipeline import RAGPipeline
from scripts.retriever_vector import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.rag = RAGPipeline(artifacts_dir)
        self.retriever = HybridRetriever(artifacts_dir)
        
    def evaluate_retrieval_only(self, eval_queries: List[Dict[str, str]], 
                               methods: List[str] = None) -> Dict[str, Any]:
        """Evaluate retrieval performance for different methods"""
        if methods is None:
            methods = ["vector", "bm25", "hybrid"]
        
        results = {}
        
        for method in methods:
            logger.info(f"🔍 Evaluating {method.upper()} retrieval...")
            
            start_time = time.time()
            metrics = self.rag.evaluate_retrieval(eval_queries, method=method, top_k=5)
            evaluation_time = time.time() - start_time
            
            results[method] = {
                **metrics,
                "evaluation_time": evaluation_time
            }
            
            logger.info(f"  Success rate: {metrics['success_rate']:.2%}")
            logger.info(f"  Avg retrieval time: {metrics['avg_retrieval_time']:.3f}s")
        
        return results
    
    def evaluate_end_to_end(self, eval_queries: List[Dict[str, str]], 
                           methods: List[str] = None, top_k: int = 5) -> Dict[str, Any]:
        """Evaluate complete RAG pipeline end-to-end"""
        if methods is None:
            methods = ["vector", "bm25", "hybrid"]
        
        results = {}
        
        for method in methods:
            logger.info(f"🚀 Evaluating {method.upper()} end-to-end...")
            
            method_results = []
            total_time = 0
            successful_generations = 0
            
            for i, query_data in enumerate(eval_queries):
                query = query_data["query"]
                logger.info(f"  Processing query {i+1}/{len(eval_queries)}: {query[:50]}...")
                
                try:
                    # Run complete RAG pipeline
                    result = self.rag.answer_query(query, method=method, top_k=top_k)
                    method_results.append(result)
                    
                    total_time += result["total_time"]
                    if result["answer"] and result["answer"] != "No relevant documents found.":
                        successful_generations += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process query '{query}': {e}")
                    method_results.append({
                        "query": query,
                        "answer": f"ERROR: {str(e)}",
                        "total_time": 0,
                        "method": method
                    })
            
            # Calculate metrics
            avg_time = total_time / len(eval_queries) if eval_queries else 0
            success_rate = successful_generations / len(eval_queries) if eval_queries else 0
            
            results[method] = {
                "method": method,
                "total_queries": len(eval_queries),
                "successful_generations": successful_generations,
                "success_rate": success_rate,
                "avg_total_time": avg_time,
                "total_time": total_time,
                "results": method_results
            }
            
            logger.info(f"  Success rate: {success_rate:.2%}")
            logger.info(f"  Avg total time: {avg_time:.3f}s")
        
        return results
    
    def compare_baseline_vs_final(self, eval_queries: List[Dict[str, str]]) -> Dict[str, Any]:
        """Compare baseline (vector only) vs final (hybrid + rerank) solution"""
        logger.info("📊 Comparing baseline vs final solution...")
        
        # Baseline: Vector retrieval only, no reranking
        logger.info("🔍 Baseline: Vector retrieval only")
        baseline_results = self.evaluate_end_to_end(
            eval_queries, methods=["vector"], top_k=5
        )
        
        # Final: Hybrid retrieval with reranking
        logger.info("🚀 Final: Hybrid retrieval with reranking")
        final_results = self.evaluate_end_to_end(
            eval_queries, methods=["hybrid"], top_k=5
        )
        
        # Calculate improvements
        baseline_metrics = baseline_results["vector"]
        final_metrics = final_results["hybrid"]
        
        improvement = {
            "success_rate_improvement": final_metrics["success_rate"] - baseline_metrics["success_rate"],
            "time_improvement": baseline_metrics["avg_total_time"] - final_metrics["avg_total_time"],
            "time_improvement_percent": (
                (baseline_metrics["avg_total_time"] - final_metrics["avg_total_time"]) / 
                baseline_metrics["avg_total_time"] * 100
            ) if baseline_metrics["avg_total_time"] > 0 else 0
        }
        
        comparison = {
            "baseline": baseline_metrics,
            "final": final_metrics,
            "improvement": improvement
        }
        
        logger.info("📈 Improvement Summary:")
        logger.info(f"  Success rate: {improvement['success_rate_improvement']:+.2%}")
        logger.info(f"  Time improvement: {improvement['time_improvement']:.3f}s")
        logger.info(f"  Time improvement: {improvement['time_improvement_percent']:+.1f}%")
        
        return comparison
    
    def generate_evaluation_report(self, eval_queries: List[Dict[str, str]], 
                                 output_dir: str = "artifacts") -> str:
        """Generate comprehensive evaluation report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info("📝 Generating comprehensive evaluation report...")
        
        # 1. Retrieval-only evaluation
        retrieval_results = self.evaluate_retrieval_only(eval_queries)
        
        # 2. End-to-end evaluation
        e2e_results = self.evaluate_end_to_end(eval_queries)
        
        # 3. Baseline vs final comparison
        comparison = self.compare_baseline_vs_final(eval_queries)
        
        # 4. Save detailed results
        report_data = {
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_queries": len(eval_queries),
            "retrieval_evaluation": retrieval_results,
            "end_to_end_evaluation": e2e_results,
            "baseline_vs_final_comparison": comparison
        }
        
        # Save JSON report
        report_file = output_path / "evaluation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # Save individual method results
        for method, results in e2e_results.items():
            method_file = output_path / f"eval_{method}.jsonl"
            self.rag.save_results(results["results"], str(method_file))
        
        # Generate summary report
        summary_file = output_path / "evaluation_summary.md"
        self._generate_summary_markdown(report_data, summary_file)
        
        logger.info(f"✅ Evaluation report saved to {output_path}")
        return str(output_path)
    
    def _generate_summary_markdown(self, report_data: Dict[str, Any], output_file: Path):
        """Generate markdown summary report"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# RAG System Evaluation Report\n\n")
            f.write(f"**Generated:** {report_data['evaluation_timestamp']}\n")
            f.write(f"**Total Queries:** {report_data['total_queries']}\n\n")
            
            # Retrieval evaluation summary
            f.write("## Retrieval Performance\n\n")
            f.write("| Method | Success Rate | Avg Time (s) |\n")
            f.write("|--------|--------------|--------------|\n")
            
            for method, metrics in report_data["retrieval_evaluation"].items():
                f.write(f"| {method.upper()} | {metrics['success_rate']:.2%} | {metrics['avg_retrieval_time']:.3f} |\n")
            
            # End-to-end evaluation summary
            f.write("\n## End-to-End Performance\n\n")
            f.write("| Method | Success Rate | Avg Total Time (s) |\n")
            f.write("|--------|--------------|-------------------|\n")
            
            for method, metrics in report_data["end_to_end_evaluation"].items():
                f.write(f"| {method.upper()} | {metrics['success_rate']:.2%} | {metrics['avg_total_time']:.3f} |\n")
            
            # Improvement summary
            improvement = report_data["baseline_vs_final_comparison"]["improvement"]
            f.write(f"\n## Improvement Summary\n\n")
            f.write(f"- **Success Rate Improvement:** {improvement['success_rate_improvement']:+.2%}\n")
            f.write(f"- **Time Improvement:** {improvement['time_improvement']:.3f}s ({improvement['time_improvement_percent']:+.1f}%)\n")

def main():
    """Main evaluation execution"""
    evaluator = RAGEvaluator()
    
    # Load evaluation queries - try comprehensive data first
    eval_file = "data/comprehensive_eval_questions.jsonl"
    if not Path(eval_file).exists():
        eval_file = "data/eval_questions.jsonl"
        print(f"⚠️ Comprehensive eval data not found, using {eval_file}")
            else:
        print(f"✅ Using comprehensive eval data: {eval_file}")
    
    if Path(eval_file).exists():
        with open(eval_file, 'r', encoding='utf-8') as f:
            eval_data = [json.loads(line) for line in f]
        
        logger.info(f"Loaded {len(eval_data)} evaluation queries")
        
        # Run comprehensive evaluation
        output_dir = evaluator.generate_evaluation_report(eval_data)
        logger.info(f"✅ Evaluation complete! Results saved to: {output_dir}")
        
    else:
        logger.error(f"Evaluation file {eval_file} not found!")
        logger.info("Please ensure you have run the preprocessing step first.")

if __name__ == "__main__":
    main()
