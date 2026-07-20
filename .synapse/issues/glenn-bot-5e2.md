---
id: glenn-bot-5e2
title: Reconsider entire implementation
status: open
type: task
priority: 2
created_at: 2026-04-12T14:15:07Z
updated_at: 2026-04-12T14:15:07Z
closed_at: ~
close_reason: ~
external_ref: gh-glenngillen/glenn-bot#24
---
I want us to rethink from first principles the best way to implement this project. My intention here was to try and approximate a digital twin of myself, by keeping track of important research, books, content, lessons, etc. that have shaped my thinking and how I approach things. For example I provide pointers to important books, or potentially upload entire PDFs/emobi formats of books, and have this system ingest all of the content. Then any given task I have I could input to my bot and it would understand the task at hand, know which bits of stored knowledge are relevant, and work out what to do next. It could be ask me a bunch of clarifying questions. It could be to suggest a specific framework to follow, it could generate a multi-step plan to execute on. This is expected to work with an incredibly broad range of topics and a vast amount of referenced content. One moment I might be asking about how to validate a new product idea, next I might mention I'm thinking of investing in a company, next I'm wanting to come up with a marketing plan to launch a new service. Each time it should be able to find all the relevant content and expertise I've saved. Find the relevant parts and lessons from it, work out how where and when I would apply those lessons, and ultimately increase the chances of me completing the task very successful. 

The current implementation was a piecemeal development that existed before it was simple to build things like agents that broke work into sub tasks, agent skills, and a number of other AI-related advancements. So I want to really think deeply about how to design such a system for massive effectiveness in my day-to-day. I think right now it's a TUI interface, but we also need to consider having an API or CLI that other applications could hook into.