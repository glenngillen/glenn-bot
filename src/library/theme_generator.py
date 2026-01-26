"""Theme generator module for AI-powered thematic categorization.

This module provides the ThemeGenerator class which uses AI (via OllamaClient)
to automatically generate broad thematic categories that group related content
across all content types in the knowledge library.

Themes are persisted to disk and can be updated incrementally as new content
is added to the knowledge base.
"""

import json
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
