"""Context detection and auto-switching for conversations."""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import json
import re
import logging
from src.ollama_client import OllamaClient

logger = logging.getLogger(__name__)


@dataclass
class ContextSwitchResult:
    """Result of context switch evaluation."""
    should_switch: bool
    recommended_context: Optional[str]
    confidence: float
    needs_confirmation: bool
    scores: Dict[str, float]


class ContextDetector:
    """Detects conversation context and recommends context switches."""

    # Default context types that can be auto-detected
    CONTEXT_TYPES = {
        "work": {
            "description": "Professional work, career, business tasks, meetings, deadlines",
            "keywords": ["work", "job", "project", "deadline", "meeting", "client", "task",
                        "report", "email", "schedule", "team", "boss", "colleague", "career",
                        "professional", "business", "company", "office", "presentation"]
        },
        "personal": {
            "description": "Personal life, family, friends, health, daily activities",
            "keywords": ["family", "friend", "health", "exercise", "home", "weekend",
                        "vacation", "hobby", "personal", "relationship", "dinner", "movie",
                        "birthday", "celebration", "shopping", "relax"]
        },
        "creative": {
            "description": "Creative projects, art, writing, music, brainstorming ideas",
            "keywords": ["creative", "write", "story", "art", "design", "music", "paint",
                        "idea", "brainstorm", "imagine", "create", "inspiration", "poem",
                        "novel", "compose", "sketch", "craft", "invent"]
        },
        "reflection": {
            "description": "Self-reflection, personal growth, philosophy, deep thinking",
            "keywords": ["reflect", "think", "feel", "growth", "mindset", "philosophy",
                        "meaning", "purpose", "goal", "dream", "fear", "hope", "believe",
                        "learn", "understand", "wisdom", "insight", "journal", "gratitude"]
        }
    }

    def __init__(self, ollama_client: OllamaClient,
                 auto_switch_threshold: float = 0.8,
                 prompt_threshold: float = 0.6):
        """Initialize the context detector.

        Args:
            ollama_client: OllamaClient for LLM-based classification
            auto_switch_threshold: Confidence above which to auto-switch silently
            prompt_threshold: Confidence above which to prompt user for confirmation
        """
        self.ollama_client = ollama_client
        self.auto_switch_threshold = auto_switch_threshold
        self.prompt_threshold = prompt_threshold

    def classify_message(self, message: str,
                        recent_context: str = None) -> Dict[str, float]:
        """Classify a message and return confidence scores for each context.

        Args:
            message: The user message to classify
            recent_context: Optional recent conversation context for better classification

        Returns:
            Dict mapping context IDs to confidence scores (0.0 to 1.0)
        """
        # First try quick keyword-based classification
        keyword_scores = self._keyword_classify(message)

        # If clear winner from keywords (high confidence), use that
        max_keyword_score = max(keyword_scores.values()) if keyword_scores else 0
        if max_keyword_score >= 0.7:
            return keyword_scores

        # Otherwise, use LLM for more nuanced classification
        try:
            llm_scores = self._llm_classify(message, recent_context)

            # Blend keyword and LLM scores (favor LLM but incorporate keywords)
            blended_scores = {}
            for context_id in self.CONTEXT_TYPES:
                keyword_score = keyword_scores.get(context_id, 0.0)
                llm_score = llm_scores.get(context_id, 0.0)
                # Weight LLM more heavily (70% LLM, 30% keywords)
                blended_scores[context_id] = (0.7 * llm_score) + (0.3 * keyword_score)

            return blended_scores

        except Exception as e:
            logger.warning(f"LLM classification failed, using keywords only: {e}")
            return keyword_scores

    def _keyword_classify(self, message: str) -> Dict[str, float]:
        """Quick keyword-based classification.

        Args:
            message: The message to classify

        Returns:
            Dict mapping context IDs to confidence scores
        """
        message_lower = message.lower()
        scores = {}

        for context_id, config in self.CONTEXT_TYPES.items():
            keyword_matches = sum(
                1 for keyword in config["keywords"]
                if keyword in message_lower
            )
            # Normalize by number of keywords (max score ~1.0 for 5+ matches)
            score = min(keyword_matches / 5.0, 1.0)
            scores[context_id] = score

        # Normalize scores so they sum to 1.0 (or close to it)
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return scores

    def _llm_classify(self, message: str,
                     recent_context: str = None) -> Dict[str, float]:
        """Use LLM for nuanced classification.

        Args:
            message: The message to classify
            recent_context: Recent conversation context

        Returns:
            Dict mapping context IDs to confidence scores
        """
        context_descriptions = "\n".join([
            f"- {context_id}: {config['description']}"
            for context_id, config in self.CONTEXT_TYPES.items()
        ])

        prompt = f"""Analyze this message and determine which conversation context it belongs to.

AVAILABLE CONTEXTS:
{context_descriptions}

MESSAGE TO CLASSIFY:
"{message}"

{f'RECENT CONVERSATION CONTEXT:{chr(10)}{recent_context}' if recent_context else ''}

Respond with ONLY a JSON object containing confidence scores (0.0 to 1.0) for each context.
The scores should reflect how well the message fits each context.
Example format:
{{"work": 0.2, "personal": 0.1, "creative": 0.6, "reflection": 0.1}}

Your response (JSON only):"""

        try:
            response = self.ollama_client.generate(
                prompt=prompt,
                system_prompt="You are a context classification assistant. Analyze messages and provide confidence scores as JSON. Be precise and consider the nuances of the message.",
                temperature=0.3  # Lower temperature for more consistent classification
            )

            # Extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                scores = json.loads(json_match.group())
                # Ensure all contexts are present
                for context_id in self.CONTEXT_TYPES:
                    if context_id not in scores:
                        scores[context_id] = 0.0
                return scores
            else:
                logger.warning(f"Could not parse LLM classification response: {response}")
                return {context_id: 0.25 for context_id in self.CONTEXT_TYPES}

        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            raise

    def should_switch_context(self, current_context_id: Optional[str],
                             scores: Dict[str, float]) -> ContextSwitchResult:
        """Determine if a context switch should occur.

        Args:
            current_context_id: ID of the current context (or None if no context)
            scores: Confidence scores from classify_message()

        Returns:
            ContextSwitchResult with recommendation
        """
        if not scores:
            return ContextSwitchResult(
                should_switch=False,
                recommended_context=None,
                confidence=0.0,
                needs_confirmation=False,
                scores={}
            )

        # Find the highest-scoring context
        best_context = max(scores, key=scores.get)
        best_score = scores[best_context]

        # If already in the best context, no switch needed
        if current_context_id == best_context:
            return ContextSwitchResult(
                should_switch=False,
                recommended_context=current_context_id,
                confidence=best_score,
                needs_confirmation=False,
                scores=scores
            )

        # Determine if we should switch
        if best_score >= self.auto_switch_threshold:
            # High confidence - auto-switch
            return ContextSwitchResult(
                should_switch=True,
                recommended_context=best_context,
                confidence=best_score,
                needs_confirmation=False,
                scores=scores
            )
        elif best_score >= self.prompt_threshold:
            # Medium confidence - ask user
            return ContextSwitchResult(
                should_switch=True,
                recommended_context=best_context,
                confidence=best_score,
                needs_confirmation=True,
                scores=scores
            )
        else:
            # Low confidence - don't switch
            return ContextSwitchResult(
                should_switch=False,
                recommended_context=None,
                confidence=best_score,
                needs_confirmation=False,
                scores=scores
            )

    def detect_and_recommend(self, message: str,
                            current_context_id: Optional[str] = None,
                            recent_context: str = None) -> ContextSwitchResult:
        """Convenience method to classify and get switch recommendation in one call.

        Args:
            message: The user message to analyze
            current_context_id: ID of the current context
            recent_context: Recent conversation context

        Returns:
            ContextSwitchResult with classification and recommendation
        """
        scores = self.classify_message(message, recent_context)
        return self.should_switch_context(current_context_id, scores)

    def get_context_description(self, context_id: str) -> str:
        """Get the description for a context type.

        Args:
            context_id: The context ID

        Returns:
            Description string or empty string if not found
        """
        if context_id in self.CONTEXT_TYPES:
            return self.CONTEXT_TYPES[context_id]["description"]
        return ""
