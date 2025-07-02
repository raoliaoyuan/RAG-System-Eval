import time
import numpy as np
from typing import List, Dict, Any, Optional
import re
import jieba

class RAGEvaluator:
    """RAG系统评估器"""
    
    def __init__(self):
        pass
        
    def evaluate_rag_response(self, 
                            query: str, 
                            answer: str, 
                            retrieved_sources: List[Dict], 
                            ground_truth: Optional[str] = None,
                            response_time: Optional[float] = None) -> Dict[str, Any]:
        """全面评估RAG响应"""
        evaluation_results = {}
        
        # 1. 答案相关性评估
        evaluation_results['answer_relevance'] = self._evaluate_answer_relevance(query, answer)
        
        # 2. 答案忠实度评估
        evaluation_results['answer_faithfulness'] = self._evaluate_answer_faithfulness(answer, retrieved_sources)
        
        # 3. 上下文精确率
        evaluation_results['context_precision'] = self._evaluate_context_precision(query, retrieved_sources)
        
        # 4. 上下文召回率
        evaluation_results['context_recall'] = self._evaluate_context_recall(query, retrieved_sources, ground_truth)
        
        # 5. 答案完整性
        evaluation_results['answer_completeness'] = self._evaluate_answer_completeness(query, answer)
        
        # 6. 答案一致性
        evaluation_results['answer_consistency'] = self._evaluate_answer_consistency(answer, retrieved_sources)
        
        # 7. 源文档多样性
        evaluation_results['source_diversity'] = self._evaluate_source_diversity(retrieved_sources)
        
        # 8. 响应性能指标
        evaluation_results['performance_metrics'] = self._evaluate_performance_metrics(
            query, answer, retrieved_sources, response_time
        )
        
        # 9. 计算综合评分
        evaluation_results['overall_score'] = self._calculate_overall_score(evaluation_results)
        
        return evaluation_results
    
    def _evaluate_answer_relevance(self, query: str, answer: str) -> Dict[str, float]:
        """评估答案与查询的相关性"""
        # 关键词匹配度
        query_keywords = set(jieba.lcut(query))
        answer_keywords = set(jieba.lcut(answer))
        keyword_overlap = len(query_keywords & answer_keywords) / max(len(query_keywords), 1)
        
        # 长度相关性
        length_ratio = min(len(answer) / max(len(query), 1), 10.0) / 10.0
        
        # 综合相关性分数
        relevance_score = (keyword_overlap + length_ratio) / 2
        
        return {
            'tfidf_similarity': float(keyword_overlap),
            'keyword_overlap': float(keyword_overlap),
            'overall_relevance': float(relevance_score)
        }
    
    def _evaluate_answer_faithfulness(self, answer: str, retrieved_sources: List[Dict]) -> Dict[str, float]:
        """评估答案对源文档的忠实度"""
        if not retrieved_sources:
            return {'faithfulness_score': 0.0, 'source_coverage': 0.0}
        
        # 计算答案与源文档的关键词重叠
        answer_keywords = set(jieba.lcut(answer))
        source_keywords = set()
        
        for source in retrieved_sources:
            source_content = source.get('content', '')
            source_keywords.update(jieba.lcut(source_content))
        
        # 忠实度分数（答案关键词在源文档中的覆盖率）
        faithfulness_score = len(answer_keywords & source_keywords) / max(len(answer_keywords), 1)
        
        # 源文档覆盖率
        source_coverage = len(answer_keywords & source_keywords) / max(len(answer_keywords), 1)
        
        return {
            'faithfulness_score': float(faithfulness_score),
            'source_coverage': float(source_coverage),
            'max_source_similarity': float(faithfulness_score)
        }
    
    def _evaluate_context_precision(self, query: str, retrieved_sources: List[Dict]) -> Dict[str, float]:
        """评估上下文精确率"""
        if not retrieved_sources:
            return {'precision_score': 0.0, 'relevant_sources_count': 0}
        
        query_keywords = set(jieba.lcut(query))
        relevant_sources = 0
        
        for source in retrieved_sources:
            source_content = source.get('content', '')
            source_keywords = set(jieba.lcut(source_content))
            
            # 如果关键词重叠超过阈值，认为是相关文档
            overlap = len(query_keywords & source_keywords) / max(len(query_keywords), 1)
            if overlap > 0.3:
                relevant_sources += 1
        
        precision_score = relevant_sources / len(retrieved_sources)
        
        return {
            'precision_score': float(precision_score),
            'avg_similarity': float(precision_score),
            'relevant_sources_count': relevant_sources,
            'total_sources_count': len(retrieved_sources)
        }
    
    def _evaluate_context_recall(self, query: str, retrieved_sources: List[Dict], ground_truth: Optional[str] = None) -> Dict[str, float]:
        """评估上下文召回率"""
        if not retrieved_sources:
            return {'recall_score': 0.0, 'coverage_estimate': 0.0}
        
        query_keywords = set(jieba.lcut(query))
        all_content = ' '.join([source.get('content', '') for source in retrieved_sources])
        content_keywords = set(jieba.lcut(all_content))
        
        keyword_coverage = len(query_keywords & content_keywords) / max(len(query_keywords), 1)
        
        return {
            'recall_score': float(keyword_coverage),
            'coverage_estimate': float(keyword_coverage),
            'keyword_coverage': float(keyword_coverage)
        }
    
    def _evaluate_answer_completeness(self, query: str, answer: str) -> Dict[str, Any]:
        """评估答案的完整性"""
        query_type = self._classify_query_type(query)
        
        if query_type == 'factual':
            completeness = self._evaluate_factual_completeness(query, answer)
        elif query_type == 'comparative':
            completeness = self._evaluate_comparative_completeness(query, answer)
        elif query_type == 'procedural':
            completeness = self._evaluate_procedural_completeness(query, answer)
        else:
            completeness = self._evaluate_general_completeness(query, answer)
        
        return {
            'completeness_score': float(completeness),
            'query_type': query_type,
            'answer_length': len(answer),
            'content_density': len(answer.strip()) / max(len(answer), 1)
        }
    
    def _evaluate_answer_consistency(self, answer: str, retrieved_sources: List[Dict]) -> Dict[str, float]:
        """评估答案的一致性"""
        if not retrieved_sources:
            return {'consistency_score': 0.0, 'contradictions_count': 0}
        
        # 简单的关键词一致性检查
        answer_keywords = set(jieba.lcut(answer))
        source_keywords = set()
        
        for source in retrieved_sources:
            source_content = source.get('content', '')
            source_keywords.update(jieba.lcut(source_content))
        
        # 一致性分数（答案关键词在源文档中的覆盖率）
        consistency_score = len(answer_keywords & source_keywords) / max(len(answer_keywords), 1)
        
        return {
            'consistency_score': float(consistency_score),
            'contradictions_count': 0,
            'consistency_checks': len(answer_keywords)
        }
    
    def _evaluate_source_diversity(self, retrieved_sources: List[Dict]) -> Dict[str, float]:
        """评估源文档的多样性"""
        if not retrieved_sources:
            return {'diversity_score': 0.0, 'unique_sources': 0}
        
        if len(retrieved_sources) < 2:
            return {'diversity_score': 1.0, 'unique_sources': 1}
        
        # 计算源文档的关键词多样性
        all_keywords = set()
        source_keyword_sets = []
        
        for source in retrieved_sources:
            source_content = source.get('content', '')
            keywords = set(jieba.lcut(source_content))
            all_keywords.update(keywords)
            source_keyword_sets.append(keywords)
        
        # 计算关键词重叠度
        overlaps = 0
        total_pairs = 0
        
        for i in range(len(source_keyword_sets)):
            for j in range(i + 1, len(source_keyword_sets)):
                overlap = len(source_keyword_sets[i] & source_keyword_sets[j]) / max(len(source_keyword_sets[i] | source_keyword_sets[j]), 1)
                overlaps += overlap
                total_pairs += 1
        
        avg_overlap = overlaps / max(total_pairs, 1)
        diversity_score = 1.0 - avg_overlap
        
        return {
            'diversity_score': float(diversity_score),
            'avg_source_similarity': float(avg_overlap),
            'unique_sources': len(set([source.get('content', '') for source in retrieved_sources]))
        }
    
    def _evaluate_performance_metrics(self, query: str, answer: str, 
                                    retrieved_sources: List[Dict], response_time: Optional[float] = None) -> Dict[str, Any]:
        """评估性能指标"""
        metrics = {
            'response_time_seconds': response_time or 0.0,
            'answer_length': len(answer),
            'sources_count': len(retrieved_sources),
            'query_length': len(query),
            'tokens_per_second': len(answer) / max(response_time, 0.1) if response_time else 0.0
        }
        
        # 计算源文档的平均相似度分数
        if retrieved_sources:
            avg_source_score = np.mean([source.get('score', 0.0) for source in retrieved_sources])
            metrics['avg_source_score'] = float(avg_source_score)
        else:
            metrics['avg_source_score'] = 0.0
        
        return metrics
    
    def _calculate_overall_score(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """计算综合评分"""
        weights = {
            'answer_relevance': 0.25,
            'answer_faithfulness': 0.20,
            'context_precision': 0.15,
            'context_recall': 0.15,
            'answer_completeness': 0.10,
            'answer_consistency': 0.10,
            'source_diversity': 0.05
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in evaluation_results:
                if isinstance(evaluation_results[metric], dict):
                    if metric == 'answer_relevance':
                        score = evaluation_results[metric]['overall_relevance']
                    elif metric == 'answer_faithfulness':
                        score = evaluation_results[metric]['faithfulness_score']
                    elif metric == 'context_precision':
                        score = evaluation_results[metric]['precision_score']
                    elif metric == 'context_recall':
                        score = evaluation_results[metric]['recall_score']
                    elif metric == 'answer_completeness':
                        score = evaluation_results[metric]['completeness_score']
                    elif metric == 'answer_consistency':
                        score = evaluation_results[metric]['consistency_score']
                    elif metric == 'source_diversity':
                        score = evaluation_results[metric]['diversity_score']
                    else:
                        continue
                else:
                    score = evaluation_results[metric]
                
                weighted_score += score * weight
                total_weight += weight
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        return {
            'overall_score': float(overall_score),
            'weighted_components': weights
        }
    
    def _classify_query_type(self, query: str) -> str:
        """分类查询类型"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['比较', '区别', '差异', 'vs', 'versus']):
            return 'comparative'
        elif any(word in query_lower for word in ['如何', '步骤', '方法', '流程']):
            return 'procedural'
        elif any(word in query_lower for word in ['是什么', '定义', '概念']):
            return 'factual'
        else:
            return 'general'
    
    def _evaluate_factual_completeness(self, query: str, answer: str) -> float:
        """评估事实性查询的完整性"""
        has_numbers = bool(re.search(r'\d+', answer))
        has_names = bool(re.search(r'[A-Za-z\u4e00-\u9fff]{2,4}', answer))
        has_details = len(answer) > 50
        
        completeness = (has_numbers + has_names + has_details) / 3
        return completeness
    
    def _evaluate_comparative_completeness(self, query: str, answer: str) -> float:
        """评估比较性查询的完整性"""
        has_comparison_words = bool(re.search(r'(比较|区别|差异|相似|不同)', answer))
        has_multiple_points = len(re.findall(r'[一二三四五六七八九十]', answer)) >= 2
        has_conclusion = len(answer) > 100
        
        completeness = (has_comparison_words + has_multiple_points + has_conclusion) / 3
        return completeness
    
    def _evaluate_procedural_completeness(self, query: str, answer: str) -> float:
        """评估程序性查询的完整性"""
        has_steps = bool(re.search(r'(步骤|方法|流程|首先|然后|最后)', answer))
        has_numbers = bool(re.search(r'\d+', answer))
        has_actions = bool(re.search(r'(做|执行|操作|使用|选择)', answer))
        
        completeness = (has_steps + has_numbers + has_actions) / 3
        return completeness
    
    def _evaluate_general_completeness(self, query: str, answer: str) -> float:
        """评估一般查询的完整性"""
        length_score = min(len(answer) / 200, 1.0)
        density_score = len(answer.strip()) / max(len(answer), 1)
        
        completeness = (length_score + density_score) / 2
        return completeness 