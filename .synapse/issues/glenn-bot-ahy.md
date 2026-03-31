---
id: glenn-bot-ahy
title: Add Response Feedback System
status: closed
type: task
priority: 2
created_at: 2025-12-23T15:22:34Z
updated_at: 2026-01-03T03:41:35Z
closed_at: 2026-01-03T03:41:35Z
close_reason: "Completed via PR #17"
external_ref: gh-glenngillen/glenn-bot#10
---
## Problem to Solve

There's no mechanism for users to provide feedback on responses. This means:
- No way to improve response quality over time
- Can't identify what types of responses work well
- No reinforcement learning from human feedback (RLHF) capability

## Current State

- Conversations are stored but not rated
- No feedback collection mechanism
- No way to use feedback to improve future responses

## Solution Proposal

1. **Feedback Collection**
   - Add thumbs up/down or 1-5 rating system
   - Store feedback alongside conversation messages
   - Collect optional text feedback for detailed improvement suggestions

2. **Feedback Storage**
   - Extend memory schema to include ratings
   - Track which responses get positive/negative feedback
   - Store correction suggestions when provided

3. **Feedback Utilization**
   - Use highly-rated responses as few-shot examples
   - Identify patterns in poorly-rated responses
   - Generate insights about preferred response styles

## Files to Modify

- `src/glenn_bot/conversation.py`: Add rating methods
- `src/glenn_bot/memory_manager.py`: Store feedback data
- `src/glenn_bot/api.py`: Add feedback endpoint

## Acceptance Criteria

- [ ] Users can rate responses
- [ ] Ratings stored with conversation context
- [ ] Can query for best/worst rated responses
- [ ] Optional: Use feedback to improve responses

## Labels

`enhancement`, `feature`

---
*Issue created via AI analysis of codebase*