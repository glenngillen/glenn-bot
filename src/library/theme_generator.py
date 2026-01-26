"""Theme generator module for AI-powered thematic categorization.

This module provides the ThemeGenerator class which uses AI (via OllamaClient)
to automatically generate broad thematic categories that group related content
across all content types in the knowledge library.

Themes are persisted to disk and can be updated incrementally as new content
is added to the knowledge base.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Union

from src.library.models import LibraryItem, Theme, ThemeAssignment

if TYPE_CHECKING:
    from src.ollama_client import OllamaClient


class ThemeGenerator:
    """AI-powered theme generator for the knowledge library.

    Uses OllamaClient to analyze content and generate thematic categories
    that group related items. Themes and assignments are persisted to JSON
    files in the data directory.

    Attributes:
        ollama_client: The OllamaClient instance for AI generation.
        data_dir: Path to the data directory for persistence.
        themes_file: Path to the themes.json file.
        assignments_file: Path to the assignments.json file.
        themes: List of Theme objects.
        assignments: List of ThemeAssignment objects.
    """

    def __init__(
        self,
        ollama_client: "OllamaClient",
        data_dir: Union[str, Path],
    ) -> None:
        """Initialize the ThemeGenerator.

        Args:
            ollama_client: The OllamaClient instance to use for AI generation.
            data_dir: Path to the data directory for storing themes and
                assignments. Will be created if it doesn't exist.
        """
        self.ollama_client = ollama_client
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # File paths for persistence
        self.themes_file = self.data_dir / "themes.json"
        self.assignments_file = self.data_dir / "assignments.json"

        # Initialize empty lists for themes and assignments
        self.themes: list[Theme] = []
        self.assignments: list[ThemeAssignment] = []

    def save_themes(self) -> None:
        """Save themes to themes.json file in the data directory.

        Serializes all themes in self.themes to JSON format and writes
        them to the themes.json file. Uses indentation for human-readable
        output. Overwrites any existing file.
        """
        themes_data = [theme.to_dict() for theme in self.themes]
        with open(self.themes_file, "w") as f:
            json.dump(themes_data, f, indent=2)

    def load_themes(self) -> list[Theme]:
        """Load themes from themes.json file in the data directory.

        Reads the themes.json file and parses it into Theme objects.
        Replaces any existing themes in self.themes with the loaded data.

        Returns:
            List of Theme objects loaded from the file. Returns empty list
            if the file doesn't exist or is empty.
        """
        if not self.themes_file.exists():
            self.themes = []
            return self.themes

        with open(self.themes_file) as f:
            themes_data = json.load(f)

        self.themes = [Theme.from_dict(data) for data in themes_data]
        return self.themes

    def save_assignments(self) -> None:
        """Save assignments to assignments.json file in the data directory.

        Serializes all assignments in self.assignments to JSON format and writes
        them to the assignments.json file. Uses indentation for human-readable
        output. Overwrites any existing file.
        """
        assignments_data = [assignment.to_dict() for assignment in self.assignments]
        with open(self.assignments_file, "w") as f:
            json.dump(assignments_data, f, indent=2)

    def load_assignments(self) -> list[ThemeAssignment]:
        """Load assignments from assignments.json file in the data directory.

        Reads the assignments.json file and parses it into ThemeAssignment objects.
        Replaces any existing assignments in self.assignments with the loaded data.

        Returns:
            List of ThemeAssignment objects loaded from the file. Returns empty list
            if the file doesn't exist or is empty.
        """
        if not self.assignments_file.exists():
            self.assignments = []
            return self.assignments

        with open(self.assignments_file) as f:
            assignments_data = json.load(f)

        self.assignments = [ThemeAssignment.from_dict(data) for data in assignments_data]
        return self.assignments

    def _build_theme_generation_prompt(self, items: list[LibraryItem]) -> str:
        """Build the prompt for LLM theme generation.

        Constructs a prompt that asks the LLM to analyze the provided content
        and generate 5-10 broad thematic categories with id, name, description,
        and keywords for each theme.

        Args:
            items: List of LibraryItem objects to analyze for theme generation.

        Returns:
            A string prompt for the LLM to generate themes in JSON format.
        """
        # Build the content summary section
        if not items:
            content_section = "No content items available for analysis."
        else:
            content_lines = []
            for item in items:
                content_type = item.content_type.value
                content_lines.append(
                    f"- [{content_type}] {item.title}: {item.summary}"
                )
            content_section = "\n".join(content_lines)

        prompt = f"""Analyze the following knowledge base content and identify 5 to 10 broad thematic categories that group related items together.

Content to analyze:
{content_section}

For each theme, provide:
- id: A URL-friendly slug (lowercase, hyphens instead of spaces, e.g., "personal-growth")
- name: A human-readable name for the theme
- description: A brief description of what this theme encompasses
- keywords: A list of representative keywords for this theme

Return your response as a JSON array of theme objects. Example format:
```json
[
  {{
    "id": "personal-growth",
    "name": "Personal Growth",
    "description": "Self-improvement, habits, and learning strategies",
    "keywords": ["growth", "habits", "learning", "self-improvement"]
  }}
]
```

Important:
- Create between 5 and 10 themes based on the content diversity
- Themes should be broad enough to group multiple items
- Each theme should have 3-6 relevant keywords
- Return only the JSON array, no additional text"""

        return prompt

    def _parse_themes_from_response(self, response: str) -> list[Theme]:
        """Parse LLM response into Theme objects.

        Extracts JSON from the LLM response (handling markdown code blocks
        and preamble text), parses it, and converts each theme object into
        a Theme dataclass instance.

        Args:
            response: The raw LLM response string, potentially containing
                JSON wrapped in markdown code blocks or with preamble text.

        Returns:
            List of Theme objects parsed from the response. Returns empty
            list if parsing fails or no valid themes are found.
        """
        if not response or not response.strip():
            return []

        # Try to extract JSON from markdown code blocks
        json_str = self._extract_json_from_response(response)
        if not json_str:
            return []

        try:
            themes_data = json.loads(json_str)
        except json.JSONDecodeError:
            return []

        if not isinstance(themes_data, list):
            return []

        themes = []
        now = datetime.now()

        for theme_data in themes_data:
            # Skip themes missing required fields
            if not isinstance(theme_data, dict):
                continue
            if "id" not in theme_data or "name" not in theme_data:
                continue

            theme = Theme(
                id=theme_data["id"],
                name=theme_data["name"],
                description=theme_data.get("description", ""),
                keywords=theme_data.get("keywords", []),
                item_count=0,
                created_at=now,
                updated_at=now,
            )
            themes.append(theme)

        return themes

    def _extract_json_from_response(self, response: str) -> str | None:
        """Extract JSON array from LLM response.

        Handles responses with markdown code blocks (```json or ```)
        and extracts the JSON content. Falls back to finding the first
        [ and last ] in the response if no code blocks are found.

        Args:
            response: The raw LLM response string.

        Returns:
            The extracted JSON string, or None if no valid JSON found.
        """
        # Try to extract from markdown code blocks first
        # Handle ```json ... ``` or ``` ... ```
        code_block_pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        matches = re.findall(code_block_pattern, response, re.DOTALL)

        if matches:
            # Return the first code block that looks like a JSON array
            for match in matches:
                stripped = match.strip()
                if stripped.startswith("["):
                    return stripped

        # Fall back to finding raw JSON array in the response
        # Find the first [ and last ]
        start_idx = response.find("[")
        end_idx = response.rfind("]")

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return response[start_idx : end_idx + 1]

        return None

    def generate_themes(self, items: list[LibraryItem]) -> list[Theme]:
        """Generate themes from library items using AI.

        Orchestrates the theme generation process by:
        1. Building a prompt from the provided items
        2. Calling OllamaClient.generate() to get AI-generated themes
        3. Parsing the response into Theme objects
        4. Saving the themes to disk
        5. Updating self.themes with the generated themes

        Args:
            items: List of LibraryItem objects to analyze for theme generation.

        Returns:
            List of Theme objects generated from the content. Returns empty
            list if generation fails or no themes could be parsed.
        """
        # Build the prompt from items
        prompt = self._build_theme_generation_prompt(items)

        # Call OllamaClient to generate themes
        system_prompt = (
            "You are a knowledge organization expert. Analyze content and identify "
            "thematic categories. Always respond with valid JSON arrays only."
        )
        response = self.ollama_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
        )

        # Parse the response into Theme objects
        themes = self._parse_themes_from_response(response)

        # Update self.themes and save to disk
        self.themes = themes
        self.save_themes()

        return themes
