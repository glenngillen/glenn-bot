"""
Feedback system for GlennBot responses.
Allows users to rate responses and provide optional text feedback.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.config import settings

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback that can be provided."""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING_1_5 = "rating_1_5"


class Rating(Enum):
    """Rating values for RATING_1_5 type."""
    TERRIBLE = 1
    POOR = 2
    OKAY = 3
    GOOD = 4
    EXCELLENT = 5


@dataclass
class Feedback:
    """Represents feedback for a single response."""
    id: str
    conversation_id: str
    message_index: int
    feedback_type: FeedbackType
    rating: int  # 1 for thumbs_down, 2 for thumbs_up, or 1-5 for rating
    text_feedback: Optional[str] = None
    timestamp: datetime = None
    user_query: Optional[str] = None  # The user's original question
    assistant_response: Optional[str] = None  # The response that was rated

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['feedback_type'] = self.feedback_type.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Feedback':
        """Create Feedback instance from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['feedback_type'] = FeedbackType(data['feedback_type'])
        return cls(**data)

    @property
    def is_positive(self) -> bool:
        """Check if the feedback is positive."""
        if self.feedback_type == FeedbackType.THUMBS_UP:
            return self.rating >= 2
        elif self.feedback_type == FeedbackType.THUMBS_DOWN:
            return False
        else:  # RATING_1_5
            return self.rating >= 4

    @property
    def normalized_score(self) -> float:
        """Get a normalized score from 0.0 to 1.0."""
        if self.feedback_type == FeedbackType.THUMBS_UP:
            return 1.0
        elif self.feedback_type == FeedbackType.THUMBS_DOWN:
            return 0.0
        else:  # RATING_1_5
            return (self.rating - 1) / 4.0  # Maps 1-5 to 0.0-1.0


class FeedbackManager:
    """Manages storage and retrieval of feedback for responses."""

    def __init__(self, feedback_dir: Optional[Path] = None):
        self.feedback_dir = feedback_dir or settings.conversation_history_dir.parent / "feedback"
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.feedback_dir / "feedback.json"
        self.feedback: Dict[str, Feedback] = {}
        self._load_feedback()

    def _load_feedback(self):
        """Load feedback from disk."""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file, 'r') as f:
                    data = json.load(f)
                self.feedback = {
                    feedback_id: Feedback.from_dict(feedback_data)
                    for feedback_id, feedback_data in data.items()
                }
                logger.info(f"Loaded {len(self.feedback)} feedback entries")
            except Exception as e:
                logger.error(f"Error loading feedback: {e}")
                self.feedback = {}

    def _save_feedback(self):
        """Save feedback to disk."""
        try:
            data = {
                feedback_id: feedback.to_dict()
                for feedback_id, feedback in self.feedback.items()
            }
            with open(self.feedback_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")

    def add_feedback(
        self,
        conversation_id: str,
        message_index: int,
        feedback_type: FeedbackType,
        rating: int,
        text_feedback: Optional[str] = None,
        user_query: Optional[str] = None,
        assistant_response: Optional[str] = None
    ) -> Feedback:
        """Add feedback for a response."""
        feedback_id = f"{conversation_id}_{message_index}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        feedback = Feedback(
            id=feedback_id,
            conversation_id=conversation_id,
            message_index=message_index,
            feedback_type=feedback_type,
            rating=rating,
            text_feedback=text_feedback,
            user_query=user_query,
            assistant_response=assistant_response
        )

        self.feedback[feedback_id] = feedback
        self._save_feedback()

        logger.info(f"Added feedback: {feedback_id} (rating={rating}, type={feedback_type.value})")
        return feedback

    def add_thumbs_up(
        self,
        conversation_id: str,
        message_index: int,
        text_feedback: Optional[str] = None,
        user_query: Optional[str] = None,
        assistant_response: Optional[str] = None
    ) -> Feedback:
        """Convenience method to add thumbs up feedback."""
        return self.add_feedback(
            conversation_id=conversation_id,
            message_index=message_index,
            feedback_type=FeedbackType.THUMBS_UP,
            rating=2,
            text_feedback=text_feedback,
            user_query=user_query,
            assistant_response=assistant_response
        )

    def add_thumbs_down(
        self,
        conversation_id: str,
        message_index: int,
        text_feedback: Optional[str] = None,
        user_query: Optional[str] = None,
        assistant_response: Optional[str] = None
    ) -> Feedback:
        """Convenience method to add thumbs down feedback."""
        return self.add_feedback(
            conversation_id=conversation_id,
            message_index=message_index,
            feedback_type=FeedbackType.THUMBS_DOWN,
            rating=1,
            text_feedback=text_feedback,
            user_query=user_query,
            assistant_response=assistant_response
        )

    def add_rating(
        self,
        conversation_id: str,
        message_index: int,
        rating: int,
        text_feedback: Optional[str] = None,
        user_query: Optional[str] = None,
        assistant_response: Optional[str] = None
    ) -> Feedback:
        """Convenience method to add 1-5 rating feedback."""
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")

        return self.add_feedback(
            conversation_id=conversation_id,
            message_index=message_index,
            feedback_type=FeedbackType.RATING_1_5,
            rating=rating,
            text_feedback=text_feedback,
            user_query=user_query,
            assistant_response=assistant_response
        )

    def get_feedback_for_message(
        self,
        conversation_id: str,
        message_index: int
    ) -> List[Feedback]:
        """Get all feedback for a specific message."""
        return [
            fb for fb in self.feedback.values()
            if fb.conversation_id == conversation_id and fb.message_index == message_index
        ]

    def get_feedback_for_conversation(self, conversation_id: str) -> List[Feedback]:
        """Get all feedback for a conversation."""
        return [
            fb for fb in self.feedback.values()
            if fb.conversation_id == conversation_id
        ]

    def get_best_responses(self, limit: int = 10) -> List[Feedback]:
        """Get the highest-rated responses."""
        positive_feedback = [
            fb for fb in self.feedback.values()
            if fb.is_positive
        ]
        return sorted(
            positive_feedback,
            key=lambda fb: (fb.normalized_score, fb.timestamp),
            reverse=True
        )[:limit]

    def get_worst_responses(self, limit: int = 10) -> List[Feedback]:
        """Get the lowest-rated responses."""
        negative_feedback = [
            fb for fb in self.feedback.values()
            if not fb.is_positive
        ]
        return sorted(
            negative_feedback,
            key=lambda fb: (fb.normalized_score, -fb.timestamp.timestamp())
        )[:limit]

    def get_responses_with_feedback_text(self, limit: int = 10) -> List[Feedback]:
        """Get responses that have text feedback (improvement suggestions)."""
        feedback_with_text = [
            fb for fb in self.feedback.values()
            if fb.text_feedback
        ]
        return sorted(
            feedback_with_text,
            key=lambda fb: fb.timestamp,
            reverse=True
        )[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        total = len(self.feedback)
        if total == 0:
            return {
                "total_feedback": 0,
                "positive_count": 0,
                "negative_count": 0,
                "average_score": 0.0,
                "feedback_with_text": 0,
                "by_type": {}
            }

        positive_count = sum(1 for fb in self.feedback.values() if fb.is_positive)
        negative_count = total - positive_count
        avg_score = sum(fb.normalized_score for fb in self.feedback.values()) / total
        feedback_with_text = sum(1 for fb in self.feedback.values() if fb.text_feedback)

        by_type = {}
        for fb in self.feedback.values():
            type_name = fb.feedback_type.value
            if type_name not in by_type:
                by_type[type_name] = {"count": 0, "avg_rating": 0.0, "total_rating": 0}
            by_type[type_name]["count"] += 1
            by_type[type_name]["total_rating"] += fb.rating

        for type_name, data in by_type.items():
            data["avg_rating"] = data["total_rating"] / data["count"]
            del data["total_rating"]

        return {
            "total_feedback": total,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_rate": positive_count / total if total > 0 else 0.0,
            "average_score": avg_score,
            "feedback_with_text": feedback_with_text,
            "by_type": by_type
        }

    def get_few_shot_examples(self, limit: int = 5) -> List[Dict[str, str]]:
        """
        Get highly-rated responses that can be used as few-shot examples.
        Returns pairs of user queries and assistant responses.
        """
        best_responses = self.get_best_responses(limit=limit * 2)  # Get extra in case some lack context

        examples = []
        for fb in best_responses:
            if fb.user_query and fb.assistant_response:
                examples.append({
                    "user_query": fb.user_query,
                    "assistant_response": fb.assistant_response,
                    "rating": fb.rating,
                    "normalized_score": fb.normalized_score
                })
                if len(examples) >= limit:
                    break

        return examples

    def get_improvement_insights(self) -> Dict[str, Any]:
        """
        Analyze feedback to identify patterns and improvement opportunities.
        """
        insights = {
            "common_issues": [],
            "successful_patterns": [],
            "improvement_suggestions": []
        }

        # Collect text feedback from negative ratings
        negative_feedback_text = [
            fb.text_feedback for fb in self.feedback.values()
            if not fb.is_positive and fb.text_feedback
        ]

        if negative_feedback_text:
            insights["common_issues"] = negative_feedback_text[:10]

        # Collect text feedback from positive ratings
        positive_feedback_text = [
            fb.text_feedback for fb in self.feedback.values()
            if fb.is_positive and fb.text_feedback
        ]

        if positive_feedback_text:
            insights["successful_patterns"] = positive_feedback_text[:10]

        # All improvement suggestions
        all_suggestions = [
            fb.text_feedback for fb in self.feedback.values()
            if fb.text_feedback
        ]

        insights["improvement_suggestions"] = all_suggestions[:20]

        return insights
