---
id: glenn-bot-3uc
title: Add Model Configuration Flexibility
status: open
type: task
priority: 2
created_at: 2025-12-23T15:22:35Z
updated_at: 2026-04-12T14:15:07Z
closed_at: ~
close_reason: ~
external_ref: gh-glenngillen/glenn-bot#9
---
## Problem to Solve

Model names and configurations are hardcoded throughout the codebase, making it difficult to switch models or adjust parameters without code changes.

## Current State

- Model names hardcoded in `glenn_agent.py` (e.g., "claude-sonnet-4-20250514")
- Embedding model hardcoded as "text-embedding-3-small"
- No configuration file for model parameters
- Temperature and other parameters not configurable

## Solution Proposal

1. **Configuration File**
   - Add `config.yaml` or use environment variables
   - Support model names, temperatures, max tokens
   - Support different configs for different use cases

2. **Model Abstraction**
   - Create model configuration class
   - Support easy switching between models (Claude, GPT, local)
   - Allow different models for different tasks (cheaper for simple, better for complex)

3. **Example Configuration**
   ```yaml
   models:
     chat:
       provider: anthropic
       model: claude-sonnet-4-20250514
       temperature: 0.7
       max_tokens: 4096
     embedding:
       provider: openai
       model: text-embedding-3-small
   ```

## Files to Modify

- `src/glenn_bot/glenn_agent.py`: Use configuration instead of hardcoded values
- `src/glenn_bot/memory_manager.py`: Configure embedding model
- Add: `config.yaml` or enhance environment variable support

## Acceptance Criteria

- [ ] Models configurable without code changes
- [ ] Support for multiple model providers
- [ ] Default configuration provided
- [ ] Documentation for configuration options

## Labels

`enhancement`, `feature`

---
*Issue created via AI analysis of codebase*