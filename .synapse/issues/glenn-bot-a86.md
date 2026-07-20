---
id: glenn-bot-a86
title: Implement LangGraph Integration
status: open
type: task
priority: 2
created_at: 2025-12-23T15:22:35Z
updated_at: 2026-04-12T14:15:07Z
closed_at: ~
close_reason: ~
external_ref: gh-glenngillen/glenn-bot#3
---
## Problem to Solve

The README mentions LangGraph as a key component of the architecture, but the actual implementation uses LangChain's `create_react_agent` instead. The codebase has LangGraph as a dependency but doesn't utilize its graph-based workflow capabilities.

## Current State

- `requirements.txt` includes `langgraph`
- `glenn_agent.py` uses `create_react_agent` from `langchain.agents`
- No graph-based state management is implemented

## Solution Proposal

Either:

### Option A: Implement LangGraph
1. Convert the agent from `create_react_agent` to a LangGraph StateGraph
2. Add proper state management for multi-step workflows
3. Implement tool nodes using LangGraph's `ToolNode`
4. Add conditional routing for different interaction modes

### Option B: Remove LangGraph Reference
1. Remove `langgraph` from `requirements.txt`
2. Update README to reflect actual architecture
3. Keep using the simpler LangChain ReAct pattern

## Acceptance Criteria

- [ ] Either fully implement LangGraph OR remove the dependency
- [ ] Update README to accurately reflect the architecture
- [ ] Update code comments to match implementation

## Labels

`enhancement`, `architecture`

---
*Issue created via AI analysis of codebase*