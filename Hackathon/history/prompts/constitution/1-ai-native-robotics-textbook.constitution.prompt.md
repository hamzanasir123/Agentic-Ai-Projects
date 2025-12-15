---
id: 1
title: AI-native Robotics Textbook Constitution
stage: constitution
date: 2025-12-15
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: 002-docusaurus-book-creation
user: user
command: /sp.constitution
labels: [constitution,ai-textbook,robotics,rag,translation]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

```text
version: 1.0

  purpose: |
    Defines rules, workflow, structure, quality standards, and
    constraints for producing a complete AI-native robotics textbook,
    RAG system, Urdu translation, and personalized dashboard.

  guiding_principles:
    - Deterministic, structured, and testable outputs.
    - No hallucinations or unstated assumptions.
    - Requirements → Tasks → Implementations must always align.
    - Maintain academic clarity & professional textbook design.
    - Modular multi-agent architecture.
    - RAG answers must be accurate with citations.
    - Urdu translation must preserve meaning and formatting.
    - All components must satisfy hackathon scoring.

  definitions:
    ai_native_book: "A Docusaurus textbook enhanced with AI assistance, RAG, personalization, and translation."
    rag_system: "Chunking + embeddings + Qdrant search + Gemini answer generation."
    agent: "An AI actor responsible for a specific function."
    subagent: "Claude Code micro-agent for file editing and code generation."
    specification: "Complete, measurable, tech-agnostic requirement document."
    done_definition: "Fully functional, validated, and ambiguity-free output."

  workflow:
    - "/sp.constitution": Defines rules and system governance.
    - "/sp.specify": Generates full functional specification.
    - "/sp.plan": Produces architecture & system diagrams.
    - "/sp.tasks": Breaks plan into concrete tasks.
    - "/sp.implement": Creates final code via Claude Code + subagents.

  quality_standards:
    specifications:
      - Must define user stories, edge cases, and success criteria.
    plans:
      - Must include diagrams, agent boundaries, and data flow.
    tasks:
      - Each task outputs a file or behavior.
      - ≤ 200 tasks.
    implementations:
      - Must follow clean architecture and avoid unused files.

  constraints:
    general:
      - No hallucinations.
      - No mixing of phases.
    frontend:
      - Must use Docusaurus.
      - Must support Urdu RTL.
    backend:
      - FastAPI + Gemini 2.5 Flash + Qdrant Cloud.
    rag:
      - Must include citations.
      - Must avoid cutting sentences during chunking.
    translation:
      - Maintain tone + semantic meaning.
    personalization:
      - Based on user metadata.
    agents:
      - Each agent must have narrow responsibility.

  acceptance_criteria:
    - All hackathon requirements met.
    - All bonus features included:
        * Subagents
        * Personalization
        * Urdu translation
        * BetterAuth
        * RAG chatbot
        * Quizzes & summaries
        * Dashboard
    - Clean repo structure.
    - Working deployment.
    - Demo video < 90 seconds.
```

## Response snapshot

I have successfully updated the project constitution based on your provided input. Here's what was accomplished:

### Changes Made:
- **Version**: Updated to 1.0 (as specified in your input)
- **Project Name**: Set to "AI-native Robotics Textbook"
- **Core Principles**: Added 6 specific principles based on your guiding principles:
  1. Deterministic and Testable Outputs
  2. Academic Clarity and Professional Design
  3. Accurate RAG with Citations
  4. Comprehensive Feature Coverage
  5. No Hallucinations and Phase Separation
  6. Narrow Agent Responsibility

- **Additional Sections**: Added "Technical Constraints and Quality Standards" and "Development Workflow and Acceptance Criteria" sections with details from your input
- **Governance**: Updated governance rules based on your requirements
- **Dates**: Set ratification and last amended dates to 2025-12-15

### Validation:
- All placeholder tokens have been replaced with concrete values
- No remaining unexplained bracket tokens
- Version line matches the sync impact report
- Dates are in ISO format (YYYY-MM-DD)
- Principles are declarative and testable

### Files Updated:
- `.specify/memory/constitution.md` - The main constitution file

### Templates Status:
- No changes needed to templates as they are generic placeholders that will be governed by the constitution principles

The constitution now fully reflects your project requirements for the AI-native robotics textbook with RAG system, Urdu translation, and all specified bonus features. The document is ready for use in governing your project development.

The constitution file has been saved at: `/Users/macbookpro/Documents/Hackathon/.specify/memory/constitution.md`

Suggested commit message: `docs: amend constitution to v1.0 (AI-native robotics textbook governance)`

## Outcome

- ✅ Impact: Constitution updated with project-specific principles and governance
- 🧪 Tests: No tests required for constitution update
- 📁 Files: 1 file updated (.specify/memory/constitution.md)
- 🔁 Next prompts: /sp.specify to generate functional specification
- 🧠 Reflection: Constitution now properly governs the AI-native robotics textbook project with all required features

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): PASS
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A