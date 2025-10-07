"""
Dawid-Skene Algorithm for consensus from multiple annotators
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
import random

logger = logging.getLogger(__name__)


class DawidSkene:
    """Dawid-Skene algorithm for consensus from multiple annotators"""
    
    def __init__(self, max_iterations: int = 100, convergence_threshold: float = 1e-6):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
    
    def fit(self, annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fit Dawid-Skene model to annotations
        
        Args:
            annotations: List of annotation dictionaries with keys:
                - 'item_id': unique identifier for the item
                - 'annotator_id': unique identifier for the annotator
                - 'label': the annotation label
                - 'confidence': optional confidence score
        
        Returns:
            Dictionary with consensus results
        """
        # Extract unique items and annotators
        items = list(set(ann['item_id'] for ann in annotations))
        annotators = list(set(ann['annotator_id'] for ann in annotations))
        labels = list(set(ann['label'] for ann in annotations))
        
        n_items = len(items)
        n_annotators = len(annotators)
        n_labels = len(label)
        
        # Create mapping dictionaries
        item_to_idx = {item: i for i, item in enumerate(items)}
        annotator_to_idx = {ann: i for i, ann in enumerate(annotators)}
        label_to_idx = {label: i for i, label in enumerate(labels)}
        
        # Initialize parameters
        # pi: true label distribution
        pi = np.ones(n_labels) / n_labels
        
        # theta: annotator confusion matrices
        theta = np.ones((n_annotators, n_labels, n_labels)) / n_labels
        
        # E-step: Compute posterior probabilities
        for iteration in range(self.max_iterations):
            old_pi = pi.copy()
            old_theta = theta.copy()
            
            # Update pi (true label distribution)
            for j in range(n_labels):
                numerator = 0
                denominator = 0
                
                for ann in annotations:
                    item_idx = item_to_idx[ann['item_id']]
                    ann_idx = annotator_to_idx[ann['annotator_id']]
                    label_idx = label_to_idx[ann['label']]
                    
                    # Weight by confidence if available
                    weight = ann.get('confidence', 1.0)
                    
                    if ann['label'] == labels[j]:
                        numerator += weight
                    denominator += weight
                
                pi[j] = numerator / max(denominator, 1e-10)
            
            # Normalize pi
            pi = pi / np.sum(pi)
            
            # Update theta (confusion matrices)
            for ann_idx, annotator in enumerate(annotators):
                for true_label_idx in range(n_labels):
                    for observed_label_idx in range(n_labels):
                        numerator = 0
                        denominator = 0
                        
                        for ann in annotations:
                            if ann['annotator_id'] == annotator:
                                item_idx = item_to_idx[ann['item_id']]
                                ann_label_idx = label_to_idx[ann['label']]
                                
                                weight = ann.get('confidence', 1.0)
                                
                                if ann_label_idx == observed_label_idx:
                                    numerator += weight * pi[true_label_idx]
                                denominator += weight * pi[true_label_idx]
                        
                        theta[ann_idx, true_label_idx, observed_label_idx] = (
                            numerator / max(denominator, 1e-10)
                        )
                
                # Normalize each row
                theta[ann_idx] = theta[ann_idx] / np.sum(theta[ann_idx], axis=1, keepdims=True)
            
            # Check convergence
            pi_diff = np.max(np.abs(pi - old_pi))
            theta_diff = np.max(np.abs(theta - old_theta))
            
            if max(pi_diff, theta_diff) < self.convergence_threshold:
                logger.info(f"Dawid-Skene converged after {iteration + 1} iterations")
                break
        
        # Compute consensus labels and confidence scores
        consensus_results = []
        for item in items:
            item_annotations = [ann for ann in annotations if ann['item_id'] == item]
            
            # Compute posterior probability for each possible label
            label_probs = {}
            for label_idx, label in enumerate(labels):
                prob = pi[label_idx]
                
                for ann in item_annotations:
                    ann_idx = annotator_to_idx[ann['annotator_id']]
                    ann_label_idx = label_to_idx[ann['label']]
                    weight = ann.get('confidence', 1.0)
                    
                    prob *= theta[ann_idx, label_idx, ann_label_idx] ** weight
                
                label_probs[label] = prob
            
            # Normalize probabilities
            total_prob = sum(label_probs.values())
            if total_prob > 0:
                label_probs = {k: v / total_prob for k, v in label_probs.items()}
            
            # Find consensus label
            consensus_label = max(label_probs.items(), key=lambda x: x[1])[0]
            confidence = label_probs[consensus_label]
            
            consensus_results.append({
                'item_id': item,
                'consensus_label': consensus_label,
                'confidence': confidence,
                'label_probabilities': label_probs,
                'annotations': item_annotations
            })
        
        return {
            'consensus_results': consensus_results,
            'pi': pi,
            'theta': theta,
            'items': items,
            'annotators': annotators,
            'labels': labels,
            'iterations': iteration + 1,
            'converged': max(pi_diff, theta_diff) < self.convergence_threshold
        }
    
    def get_annotator_quality(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Get annotator quality scores from fitted model"""
        theta = result['theta']
        annotators = result['annotators']
        
        quality_scores = {}
        for i, annotator in enumerate(annotators):
            # Quality score is the average diagonal of confusion matrix
            quality_scores[annotator] = np.mean(np.diag(theta[i]))
        
        return quality_scores
