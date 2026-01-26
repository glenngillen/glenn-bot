# Implementation Plan: Browsable Knowledge Library

## Summary

Implement a static website generator that creates a browsable library of all knowledge base content from ChromaDB. The library will feature AI-generated thematic categorization, book cover resolution via APIs, client-side search, and a minimal card-based UI. The feature integrates with the existing CLI as `/build-library` and `/serve-library` commands.

## Gap Analysis

### Existing Components (Verified)
- `KnowledgeBase` class with `export_knowledge()` method (src/knowledge_base.py:191-253)
  - Returns: `export_version`, `export_timestamp`, `collection_name`, `total_documents`, `filters`, `documents`
  - Documents include: `id`, `content`, `metadata` (with `type`, `name`, `source`, optional `category`), `embedding`
- `OllamaClient.generate()` for LLM calls (src/ollama_client.py:14-41)
  - Accepts: `prompt`, `context`, `system_prompt`, `temperature`
- CLI command handler in `GlennBot.handle_command()` (src/main.py:69-268)
- Rich terminal UI with `show_thinking_indicator()` context manager (src/terminal_ui.py:195-197)
- `TerminalUI.display_help()` at line 126 - needs library commands section added
- Test fixtures: `mock_ollama_client`, `mock_knowledge_base`, `temp_dir`, `mock_settings` (tests/conftest.py)
- `Settings` class using pydantic-settings (src/config.py:4-22)
- `requests` library already in requirements.txt (for cover image API calls)

### Document Types in ChromaDB
| Type | Description |
|------|-------------|
| value | Core values and principles |
| framework | Problem-solving frameworks |
| preference | User preferences |
| memory | Conversation memories |
| web | Web-scraped content |

### Missing Components (To Implement)
- Entire `src/library/` module (8 Python files)
- Jinja2 templates in `src/library/templates/` (6 templates)
- Static assets in `src/library/assets/` (CSS, JS, placeholder images)
- Jinja2 dependency in `requirements.txt`
- CLI commands: `/build-library`, `/serve-library`, `/library-status`
- Library settings in `src/config.py`
- Library test fixtures in `tests/conftest.py`
- Data directories: `data/library/`, `data/library-site/`

## Architecture

```
src/library/
├── __init__.py
├── models.py              # LibraryItem, Theme, ContentType, ThemeAssignment
├── content_exporter.py    # Export ChromaDB content to LibraryItems
├── theme_generator.py     # AI-powered theme generation
├── cover_resolver.py      # Book cover API lookups + placeholders
├── search_indexer.py      # Build client-side search index
├── static_generator.py    # Generate HTML pages with Jinja2
├── server.py              # Simple HTTP server
└── builder.py             # Orchestrates build process

src/library/templates/
├── base.html              # Base layout
├── index.html             # Home page
├── theme.html             # Theme page
├── item.html              # Item detail page
├── all.html               # All content page
└── search.html            # Search results page

src/library/assets/
├── css/styles.css
├── js/search.js
└── images/placeholders/   # 10 SVG icons by content type

data/library/              # Persistent data
├── themes.json
├── assignments.json
└── cover_cache.json

data/library-site/         # Generated site
├── index.html
├── all/index.html
├── theme/{id}/index.html
├── item/{id}/index.html
├── search/index.html
├── assets/
└── _data/library.json
```

## Tasks

### Phase 1: Project Setup & Dependencies (6 tasks)

- [x] 1. Add Jinja2 dependency to requirements.txt
- [x] 2. Create `src/library/` package with `__init__.py`
- [x] 3. Add library settings to `src/config.py`:
  - `library_data_dir: Path = Path("./data/library")`
  - `library_site_dir: Path = Path("./data/library-site")`
  - `library_server_port: int = 8080`
- [ ] 4. Create `data/library/` directory structure (themes, cache, assignments)
- [ ] 5. Add library-specific test fixtures to `tests/conftest.py`:
  - `sample_library_item` fixture
  - `sample_theme` fixture
  - `sample_theme_assignment` fixture
  - `sample_chromadb_document` fixture
- [ ] 6. Create test file `tests/test_library_models.py`

### Phase 2: Core Data Models (12 tasks)

- [x] 7. Write tests for ContentType enum (10 types: BOOK, ARTICLE, FRAMEWORK, VALUE, PREFERENCE, MEMORY, INSIGHT, GOAL, SKILL, WEB_CONTENT)
- [x] 8. Implement ContentType enum in `src/library/models.py`
- [x] 9. Write tests for LibraryItem dataclass (id, content_type, title, summary, full_content, source_url, cover_image_url, metadata, themes, created_at, highlights)
- [x] 10. Implement LibraryItem dataclass with all fields
- [x] 11. Write tests for LibraryItem.to_dict() and LibraryItem.from_dict() serialization
- [x] 12. Implement LibraryItem serialization methods
- [x] 13. Write tests for Theme dataclass (id, name, description, keywords, item_count, created_at, updated_at)
- [x] 14. Implement Theme dataclass in `src/library/models.py`
- [x] 15. Write tests for Theme.to_dict() and Theme.from_dict() serialization
- [x] 16. Implement Theme serialization methods
- [x] 17. Write tests for ThemeAssignment dataclass (item_id, theme_id, confidence, assigned_at)
- [x] 18. Implement ThemeAssignment dataclass with serialization

### Phase 3: Content Export (15 tasks)

- [x] 19. Create test file `tests/test_content_exporter.py`
- [x] 20. Write tests for ContentExporter initialization with KnowledgeBase
- [x] 21. Implement ContentExporter class skeleton in `src/library/content_exporter.py`
- [x] 22. Write tests for _infer_content_type() mapping (value->VALUE, framework->FRAMEWORK, web->WEB_CONTENT, preference->PREFERENCE, memory->MEMORY)
- [x] 23. Implement _infer_content_type() content type mapping
- [x] 24. Write tests for _generate_title() when metadata lacks 'name' (first 50 chars)
- [x] 25. Implement _generate_title() title generation
- [x] 26. Write tests for _generate_summary() truncation (200 char limit with ellipsis)
- [x] 27. Implement _generate_summary() summary truncation logic
- [x] 28. Write tests for _extract_highlights() (key_points from values, steps from frameworks)
- [x] 29. Implement _extract_highlights() extraction by content type
- [x] 30. Write tests for _convert_document() transforming ChromaDB doc to LibraryItem
- [x] 31. Implement _convert_document() method
- [x] 32. Write tests for export_all() returning List[LibraryItem]
- [x] 33. Implement export_all() calling KnowledgeBase.export_knowledge()

### Phase 4: Theme Generation (29 tasks)

- [x] 34. Create test file `tests/test_theme_generator.py`
- [x] 35. Write tests for ThemeGenerator initialization with OllamaClient and data_dir
- [x] 36. Implement ThemeGenerator class skeleton in `src/library/theme_generator.py`
- [x] 37. Write tests for save_themes() persisting to themes.json
- [x] 38. Implement save_themes() method
- [x] 39. Write tests for load_themes() reading from themes.json
- [x] 40. Implement load_themes() method
- [x] 41. Write tests for save_assignments() persisting to assignments.json
- [x] 42. Implement save_assignments() method
- [x] 43. Write tests for load_assignments() reading from assignments.json
- [x] 44. Implement load_assignments() method
- [x] 45. Write tests for _build_theme_generation_prompt() constructing LLM prompt
- [x] 46. Implement _build_theme_generation_prompt() method
- [x] 47. Write tests for _parse_themes_from_response() parsing JSON into Theme objects
- [x] 48. Implement _parse_themes_from_response() method
- [x] 49. Write tests for generate_themes() orchestration (LLM call, parse, save)
- [x] 50. Implement generate_themes() calling OllamaClient.generate()
- [x] 51. Write tests for _build_assignment_prompt() for item-to-theme assignment
- [x] 52. Implement _build_assignment_prompt() method
- [ ] 53. Write tests for _parse_assignments_from_response() parsing into ThemeAssignment
- [ ] 54. Implement _parse_assignments_from_response() method
- [ ] 55. Write tests for assign_items_to_themes() with confidence scores
- [ ] 56. Implement assign_items_to_themes() method
- [ ] 57. Write tests for update_assignments() for incremental updates
- [ ] 58. Implement update_assignments() for new content
- [ ] 59. Write tests for "Miscellaneous" catch-all theme (confidence < 0.3)
- [ ] 60. Implement catch-all theme handling
- [ ] 61. Write tests for get_items_for_theme() returning assigned items
- [ ] 62. Implement get_items_for_theme() method

### Phase 5: Cover Image Resolution (21 tasks)

- [ ] 63. Create test file `tests/test_cover_resolver.py`
- [ ] 64. Write tests for CoverResolver initialization with cache_dir
- [ ] 65. Implement CoverResolver class skeleton in `src/library/cover_resolver.py`
- [ ] 66. Write tests for _load_cache() reading cover_cache.json
- [ ] 67. Implement _load_cache() method
- [ ] 68. Write tests for _save_cache() persisting cover_cache.json
- [ ] 69. Implement _save_cache() method
- [ ] 70. Write tests for _fetch_cover_by_isbn() calling Open Library API
- [ ] 71. Implement _fetch_cover_by_isbn() using requests library
- [ ] 72. Write tests for _fetch_cover_by_title() calling Open Library API
- [ ] 73. Implement _fetch_cover_by_title() fallback
- [ ] 74. Write tests for _fetch_cover_from_google_books() API call
- [ ] 75. Implement _fetch_cover_from_google_books() fallback
- [ ] 76. Write tests for get_placeholder_url() (10 distinct placeholders)
- [ ] 77. Implement get_placeholder_url() method
- [ ] 78. Write tests for resolve_cover() orchestration (cache -> API -> placeholder)
- [ ] 79. Implement resolve_cover() with fallback chain
- [ ] 80. Write tests for resolve_all_covers() batch processing
- [ ] 81. Implement resolve_all_covers() method
- [ ] 82. Write tests for API error handling with exponential backoff
- [ ] 83. Implement exponential backoff and graceful degradation

### Phase 6: Search Index Generation (11 tasks)

- [ ] 84. Create test file `tests/test_search_indexer.py`
- [ ] 85. Write tests for SearchIndexer initialization
- [ ] 86. Implement SearchIndexer class in `src/library/search_indexer.py`
- [ ] 87. Write tests for _extract_keywords() from themes and content
- [ ] 88. Implement _extract_keywords() method
- [ ] 89. Write tests for _build_search_item() creating index entry
- [ ] 90. Implement _build_search_item() method
- [ ] 91. Write tests for build_index() returning search items
- [ ] 92. Implement build_index() method
- [ ] 93. Write tests for write_index() outputting search-index.json
- [ ] 94. Implement write_index() to generate search-index.json

### Phase 7: Static Assets (11 tasks)

- [ ] 95. Create `src/library/assets/css/` directory
- [ ] 96. Implement styles.css with design system:
  - Background: #FAFAFA, Cards: #FFFFFF
  - Text: #1A1A1A (primary), #666666 (secondary)
  - Accent: #2563EB
- [ ] 97. Implement responsive grid CSS (4-col @1024px+, 3-col @768px, 1-2 col <768px)
- [ ] 98. Implement badge colors for 10 content types
- [ ] 99. Implement highlight/blockquote styling (border-left 4px accent)
- [ ] 100. Create `src/library/assets/js/` directory
- [ ] 101. Implement search.js with Fuse.js CDN (lazy-load, 300ms debounce)
- [ ] 102. Implement sort functionality (Recent, A-Z, By Type)
- [ ] 103. Implement pagination/load-more (50 items per page)
- [ ] 104. Create `src/library/assets/images/placeholders/` directory
- [ ] 105. Create 10 SVG placeholder icons (book, article, framework, value, preference, memory, insight, goal, skill, web_content)

### Phase 8: HTML Templates (8 tasks)

- [ ] 106. Create `src/library/templates/` directory
- [ ] 107. Implement base.html (header nav, footer, CSS/JS includes, Fuse.js CDN)
- [ ] 108. Implement _card.html macro (cover, badge, title, summary, link)
- [ ] 109. Implement index.html (home with theme cards, stats, link to all)
- [ ] 110. Implement theme.html (name, description, breadcrumb, item grid)
- [ ] 111. Implement item.html (cover, title, badge, summary, source link, highlights, theme badges, metadata)
- [ ] 112. Implement all.html (sort dropdown, item grid, pagination)
- [ ] 113. Implement search.html (search input, results, no-results state)

### Phase 9: Static Site Generator (21 tasks)

- [ ] 114. Create test file `tests/test_static_generator.py`
- [ ] 115. Write tests for StaticGenerator initialization
- [ ] 116. Implement StaticGenerator class in `src/library/static_generator.py`
- [ ] 117. Write tests for _setup_jinja_env() loading templates
- [ ] 118. Implement _setup_jinja_env() method
- [ ] 119. Write tests for _ensure_output_dirs() creating structure
- [ ] 120. Implement _ensure_output_dirs() method
- [ ] 121. Write tests for generate_home_page() rendering index.html
- [ ] 122. Implement generate_home_page() method
- [ ] 123. Write tests for generate_theme_pages() creating theme/{id}/index.html
- [ ] 124. Implement generate_theme_pages() method
- [ ] 125. Write tests for generate_item_pages() creating item/{id}/index.html
- [ ] 126. Implement generate_item_pages() method
- [ ] 127. Write tests for generate_all_page() rendering all/index.html
- [ ] 128. Implement generate_all_page() method
- [ ] 129. Write tests for generate_search_page() rendering search/index.html
- [ ] 130. Implement generate_search_page() method
- [ ] 131. Write tests for copy_assets() copying CSS, JS, images
- [ ] 132. Implement copy_assets() method
- [ ] 133. Write tests for generate_all() orchestrating full generation
- [ ] 134. Implement generate_all() method

### Phase 10: Build Process & Caching (17 tasks)

- [ ] 135. Create test file `tests/test_library_builder.py`
- [ ] 136. Write tests for LibraryBuilder initialization
- [ ] 137. Implement LibraryBuilder class in `src/library/builder.py`
- [ ] 138. Write tests for _load_build_state() reading _build_state.json
- [ ] 139. Implement _load_build_state() method
- [ ] 140. Write tests for _save_build_state() persisting state
- [ ] 141. Implement _save_build_state() method
- [ ] 142. Write tests for _compute_item_hash() generating content hash
- [ ] 143. Implement _compute_item_hash() method
- [ ] 144. Write tests for _get_changed_items() detecting modifications
- [ ] 145. Implement _get_changed_items() method
- [ ] 146. Write tests for build() orchestration (export -> covers -> themes -> index -> pages)
- [ ] 147. Implement build() method
- [ ] 148. Write tests for build() with force=True
- [ ] 149. Implement force flag handling
- [ ] 150. Write tests for _export_library_json() creating debug file
- [ ] 151. Implement _export_library_json() method

### Phase 11: Local Server (7 tasks)

- [ ] 152. Create test file `tests/test_library_server.py`
- [ ] 153. Write tests for LibraryServer initialization
- [ ] 154. Implement LibraryServer class in `src/library/server.py`
- [ ] 155. Write tests for serve() starting HTTP server
- [ ] 156. Implement serve() using http.server
- [ ] 157. Write tests for port-in-use handling
- [ ] 158. Implement port fallback logic

### Phase 12: CLI Integration (14 tasks)

- [ ] 159. Write tests for /build-library command handler
- [ ] 160. Implement _build_library() method in GlennBot
- [ ] 161. Add /build-library to handle_command() in src/main.py
- [ ] 162. Write tests for /build-library --serve flag
- [ ] 163. Implement --serve flag handling
- [ ] 164. Write tests for /build-library --force flag
- [ ] 165. Implement --force flag handling
- [ ] 166. Write tests for /serve-library command
- [ ] 167. Implement _serve_library() method
- [ ] 168. Add /serve-library to handle_command()
- [ ] 169. Write tests for /library-status command
- [ ] 170. Implement _show_library_status() method
- [ ] 171. Add /library-status to handle_command()
- [ ] 172. Update display_help() in src/terminal_ui.py with library commands section

### Phase 13: Edge Cases & Polish (14 tasks)

- [ ] 173. Write tests for empty knowledge base handling
- [ ] 174. Implement "No content yet" message in templates
- [ ] 175. Write tests for long content truncation
- [ ] 176. Implement CSS truncation with ellipsis
- [ ] 177. Write tests for special character escaping
- [ ] 178. Verify Jinja2 autoescape is enabled
- [ ] 179. Write tests for items without source_url
- [ ] 180. Implement conditional source link in item.html
- [ ] 181. Write tests for items without highlights
- [ ] 182. Implement conditional highlights section
- [ ] 183. Write tests for small knowledge base (minimum 3 themes)
- [ ] 184. Implement minimum theme count logic
- [ ] 185. Write integration test for full build process
- [ ] 186. Write integration test for valid HTML output

## Notes

### Dependencies
- `Jinja2>=3.0.0` - Template rendering (add to requirements.txt)

### Key Integration Points
- `KnowledgeBase.export_knowledge()` at src/knowledge_base.py:191 - Source of all content
- `OllamaClient.generate()` at src/ollama_client.py:14 - Theme generation and assignment
- `GlennBot.handle_command()` at src/main.py:69 - CLI command registration
- `TerminalUI.show_thinking_indicator()` at src/terminal_ui.py:195 - Progress display
- `TerminalUI.display_help()` at src/terminal_ui.py:126 - Help text update

### Storage Locations
| Data | Location |
|------|----------|
| Themes | `data/library/themes.json` |
| Assignments | `data/library/assignments.json` |
| Cover cache | `data/library/cover_cache.json` |
| Build output | `data/library-site/` |
| Build state | `data/library-site/_build_state.json` |

### Content Type Mapping
| ChromaDB | LibraryItem |
|----------|-------------|
| value | VALUE |
| framework | FRAMEWORK |
| preference | PREFERENCE |
| memory | MEMORY |
| web | WEB_CONTENT |
| book | BOOK |
| article | ARTICLE |
| insight | INSIGHT |
| goal | GOAL |
| skill | SKILL |

### Badge Colors
| Type | Color |
|------|-------|
| book | #3B82F6 (blue) |
| article | #10B981 (green) |
| framework | #8B5CF6 (purple) |
| value | #EC4899 (pink) |
| insight | #F59E0B (amber) |
| memory | #6366F1 (indigo) |
| goal | #14B8A6 (teal) |
| skill | #F97316 (orange) |
| web_content | #64748B (slate) |
| preference | #EF4444 (red) |

### API Endpoints
- Open Library ISBN: `https://covers.openlibrary.org/b/isbn/{ISBN}-L.jpg`
- Open Library title: `https://covers.openlibrary.org/b/title/{TITLE}-L.jpg`
- Google Books: `https://www.googleapis.com/books/v1/volumes?q=intitle:{TITLE}`

### CLI Commands
```
/build-library              # Full build
/build-library --serve      # Build and serve
/build-library --force      # Force regenerate
/serve-library              # Serve existing build
/library-status             # Show build stats
```

### Test Patterns
```python
@patch('src.library.module.OllamaClient')
@patch('src.library.module.settings')
def test_feature(self, mock_settings, mock_ollama_class, temp_dir):
    mock_settings.library_data_dir = temp_dir / "library"
    mock_settings.library_site_dir = temp_dir / "library-site"

    mock_ollama = MagicMock()
    mock_ollama.generate.return_value = '{"themes": [...]}'
    mock_ollama_class.return_value = mock_ollama
```

### TerminalUI Integration
```python
with self.ui.show_thinking_indicator():
    result = builder.build()
self.ui.console.print(f"[green]Build complete![/green]")
```

### Task Summary
| Phase | Tasks | Description |
|-------|-------|-------------|
| 1 | 6 | Setup & Dependencies |
| 2 | 12 | Core Data Models |
| 3 | 15 | Content Export |
| 4 | 29 | Theme Generation |
| 5 | 21 | Cover Images |
| 6 | 11 | Search Index |
| 7 | 11 | Static Assets |
| 8 | 8 | HTML Templates |
| 9 | 21 | Static Generator |
| 10 | 17 | Build Process |
| 11 | 7 | Local Server |
| 12 | 14 | CLI Integration |
| 13 | 14 | Edge Cases |
| **Total** | **186** | |
