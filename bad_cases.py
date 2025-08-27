import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from collections import defaultdict, Counter
import re
from scripts.rag_pipeline import RAGPipeline
from scripts.retriever_vector import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorCaseAnalyzer:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.rag = RAGPipeline(artifacts_dir)
        self.retriever = HybridRetriever(artifacts_dir)
        
    def analyze_evaluation_results(self, eval_files: List[str] = None) -> Dict[str, Any]:
        """Analyze evaluation results to identify error cases"""
        if eval_files is None:
            eval_files = [
                "artifacts/eval_vector.jsonl",
                "artifacts/eval_bm25.jsonl", 
                "artifacts/eval_hybrid.jsonl"
            ]
        
        error_cases = []
        performance_issues = []
        
        for eval_file in eval_files:
            file_path = Path(eval_file)
            if not file_path.exists():
                logger.warning(f"Evaluation file not found: {eval_file}")
                continue
            
            logger.info(f"🔍 Analyzing {eval_file}...")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    result = json.loads(line)
                    analysis = self._analyze_single_result(result)
                    
                    if analysis["has_errors"]:
                        error_cases.append(analysis)
                    
                    if analysis["performance_issues"]:
                        performance_issues.append(analysis)
        
        return {
            "error_cases": error_cases,
            "performance_issues": performance_issues,
            "total_analyzed": len(error_cases) + len(performance_issues)
        }
    
    def _analyze_single_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single evaluation result"""
        analysis = {
            "query": result.get("query", ""),
            "method": result.get("method", ""),
            "has_errors": False,
            "error_types": [],
            "performance_issues": [],
            "retrieval_quality": "unknown",
            "generation_quality": "unknown",
            "recommendations": []
        }
        
        # Check for errors
        if "ERROR:" in result.get("answer", ""):
            analysis["has_errors"] = True
            analysis["error_types"].append("generation_error")
            analysis["recommendations"].append("Check model loading and generation pipeline")
        
        # Check for no documents found
        if result.get("answer") == "No relevant documents found.":
            analysis["has_errors"] = True
            analysis["error_types"].append("retrieval_failure")
            analysis["recommendations"].append("Improve retrieval strategy or expand knowledge base")
        
        # Check for empty answers
        if not result.get("answer") or result.get("answer").strip() == "":
            analysis["has_errors"] = True
            analysis["error_types"].append("empty_response")
            analysis["recommendations"].append("Check generation parameters and prompt engineering")
        
        # Check performance issues
        if result.get("total_time", 0) > 10.0:  # More than 10 seconds
            analysis["performance_issues"].append("slow_response")
            analysis["recommendations"].append("Optimize retrieval and generation pipeline")
        
        if result.get("retrieval_time", 0) > 5.0:  # More than 5 seconds
            analysis["performance_issues"].append("slow_retrieval")
            analysis["recommendations"].append("Optimize vector search and indexing")
        
        if result.get("generation_time", 0) > 5.0:  # More than 5 seconds
            analysis["performance_issues"].append("slow_generation")
            analysis["recommendations"].append("Consider model optimization or hardware upgrade")
        
        # Analyze retrieval quality
        retrieved_docs = result.get("retrieved_docs", [])
        if retrieved_docs:
            analysis["retrieval_quality"] = self._assess_retrieval_quality(
                result["query"], retrieved_docs
            )
        
        # Analyze generation quality
        if result.get("answer") and result.get("answer") != "No relevant documents found.":
            analysis["generation_quality"] = self._assess_generation_quality(
                result["query"], result["answer"], result.get("context", "")
            )
        
        return analysis
    
    def _assess_retrieval_quality(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Assess the quality of retrieved documents"""
        if not retrieved_docs:
            return "no_documents"
        
        # Check if any document contains query keywords
        query_keywords = set(re.findall(r'\w+', query.lower()))
        
        relevant_docs = 0
        for doc in retrieved_docs:
            chunk_text = doc["chunk"]["chunk"].lower()
            chunk_keywords = set(re.findall(r'\w+', chunk_text))
            
            # Calculate keyword overlap
            overlap = len(query_keywords.intersection(chunk_keywords))
            if overlap > 0:
                relevant_docs += 1
        
        relevance_ratio = relevant_docs / len(retrieved_docs)
        
        if relevance_ratio >= 0.8:
            return "excellent"
        elif relevance_ratio >= 0.6:
            return "good"
        elif relevance_ratio >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _assess_generation_quality(self, query: str, answer: str, context: str) -> str:
        """Assess the quality of generated answers"""
        if not answer or answer == "No relevant documents found.":
            return "no_answer"
        
        # Check answer length
        if len(answer) < 10:
            return "too_short"
        
        if len(answer) > 500:
            return "too_long"
        
        # Check if answer contains context information
        if context and not any(word.lower() in answer.lower() for word in context.split()[:10]):
            return "context_mismatch"
        
        # Check for repetitive text
        words = answer.split()
        if len(words) > 20:
            word_counts = Counter(words)
            if max(word_counts.values()) > len(words) * 0.3:  # More than 30% repetition
                return "repetitive"
        
        return "good"
    
    def identify_common_patterns(self, error_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify common error patterns"""
        patterns = {
            "error_types": Counter(),
            "retrieval_quality": Counter(),
            "generation_quality": Counter(),
            "methods_with_issues": Counter(),
            "common_keywords": Counter()
        }
        
        for case in error_cases:
            # Count error types
            for error_type in case["error_types"]:
                patterns["error_types"][error_type] += 1
            
            # Count retrieval quality
            patterns["retrieval_quality"][case["retrieval_quality"]] += 1
            
            # Count generation quality
            patterns["generation_quality"][case["generation_quality"]] += 1
            
            # Count methods with issues
            patterns["methods_with_issues"][case["method"]] += 1
            
            # Extract common keywords from problematic queries
            query_words = re.findall(r'\w+', case["query"].lower())
            for word in query_words:
                if len(word) > 3:  # Only meaningful words
                    patterns["common_keywords"][word] += 1
        
        return patterns
    
    def generate_improvement_plan(self, patterns: Dict[str, Any], 
                                 error_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate improvement plan based on error analysis"""
        improvement_plan = {
            "priority_issues": [],
            "retrieval_improvements": [],
            "generation_improvements": [],
            "system_optimizations": [],
            "monitoring_recommendations": []
        }
        
        # Identify priority issues
        most_common_errors = patterns["error_types"].most_common(3)
        for error_type, count in most_common_errors:
            if error_type == "retrieval_failure":
                improvement_plan["priority_issues"].append({
                    "issue": "High retrieval failure rate",
                    "impact": "High",
                    "recommendation": "Improve document indexing and retrieval strategy",
                    "estimated_effort": "Medium"
                })
            elif error_type == "generation_error":
                improvement_plan["priority_issues"].append({
                    "issue": "Generation pipeline errors",
                    "impact": "High",
                    "recommendation": "Fix model loading and generation pipeline",
                    "estimated_effort": "High"
                })
        
        # Retrieval improvements
        if patterns["retrieval_quality"]["poor"] > 0:
            improvement_plan["retrieval_improvements"].extend([
                "Implement better chunking strategy",
                "Add semantic similarity thresholds",
                "Consider query expansion techniques",
                "Implement feedback loop for retrieval quality"
            ])
        
        # Generation improvements
        if patterns["generation_quality"]["context_mismatch"] > 0:
            improvement_plan["generation_improvements"].extend([
                "Improve prompt engineering",
                "Add context validation",
                "Implement answer quality checks",
                "Consider fine-tuning on domain-specific data"
            ])
        
        # System optimizations
        if patterns["error_types"]["slow_response"] > 0:
            improvement_plan["system_optimizations"].extend([
                "Optimize vector search algorithms",
                "Implement caching for frequent queries",
                "Consider model quantization",
                "Optimize batch processing"
            ])
        
        # Monitoring recommendations
        improvement_plan["monitoring_recommendations"].extend([
            "Implement real-time error tracking",
            "Add performance monitoring dashboards",
            "Set up alerting for error rate thresholds",
            "Track user satisfaction metrics"
        ])
        
        return improvement_plan
    
    def generate_error_report(self, output_dir: str = "artifacts") -> str:
        """Generate comprehensive error analysis report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info("📝 Generating error analysis report...")
        
        # Analyze evaluation results
        analysis_results = self.analyze_evaluation_results()
        
        # Identify patterns
        patterns = self.identify_common_patterns(analysis_results["error_cases"])
        
        # Generate improvement plan
        improvement_plan = self.generate_improvement_plan(patterns, analysis_results["error_cases"])
        
        # Compile full report
        report_data = {
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_error_cases": len(analysis_results["error_cases"]),
                "total_performance_issues": len(analysis_results["performance_issues"]),
                "most_common_error": patterns["error_types"].most_common(1)[0] if patterns["error_types"] else None
            },
            "error_patterns": patterns,
            "error_cases": analysis_results["error_cases"],
            "performance_issues": analysis_results["performance_issues"],
            "improvement_plan": improvement_plan
        }
        
        # Save detailed report
        report_file = output_path / "error_analysis_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # Save error cases for further analysis
        error_cases_file = output_path / "bad_cases.jsonl"
        with open(error_cases_file, 'w', encoding='utf-8') as f:
            for case in analysis_results["error_cases"]:
                f.write(json.dumps(case, ensure_ascii=False) + '\n')
        
        # Generate markdown summary
        summary_file = output_path / "error_analysis_summary.md"
        self._generate_markdown_summary(report_data, summary_file)
        
        logger.info(f"✅ Error analysis report saved to {output_path}")
        return str(output_path)
    
    def _generate_markdown_summary(self, report_data: Dict[str, Any], output_file: Path):
        """Generate markdown summary of error analysis"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Error Case Analysis Report\n\n")
            f.write(f"**Generated:** {report_data['analysis_timestamp']}\n\n")
            
            # Summary
            summary = report_data["summary"]
            f.write("## Summary\n\n")
            f.write(f"- **Total Error Cases:** {summary['total_error_cases']}\n")
            f.write(f"- **Performance Issues:** {summary['total_performance_issues']}\n")
            if summary['most_common_error']:
                f.write(f"- **Most Common Error:** {summary['most_common_error'][0]} ({summary['most_common_error'][1]} occurrences)\n")
            f.write("\n")
            
            # Error patterns
            f.write("## Error Patterns\n\n")
            f.write("### Error Types\n")
            for error_type, count in report_data["error_patterns"]["error_types"].most_common():
                f.write(f"- {error_type}: {count} occurrences\n")
            f.write("\n")
            
            # Improvement plan
            f.write("## Improvement Plan\n\n")
            
            f.write("### Priority Issues\n")
            for issue in report_data["improvement_plan"]["priority_issues"]:
                f.write(f"- **{issue['issue']}** (Impact: {issue['impact']}, Effort: {issue['estimated_effort']})\n")
                f.write(f"  - {issue['recommendation']}\n")
            f.write("\n")
            
            f.write("### Retrieval Improvements\n")
            for improvement in report_data["improvement_plan"]["retrieval_improvements"]:
                f.write(f"- {improvement}\n")
            f.write("\n")
            
            f.write("### Generation Improvements\n")
            for improvement in report_data["improvement_plan"]["generation_improvements"]:
                f.write(f"- {improvement}\n")
            f.write("\n")

def main():
    """Main function to run error analysis"""
    analyzer = ErrorCaseAnalyzer()
    
    # Generate error analysis report
    output_dir = analyzer.generate_error_report()
    logger.info(f"✅ Error analysis complete! Report saved to: {output_dir}")

if __name__ == "__main__":
    main()
