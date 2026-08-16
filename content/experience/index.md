---
title: "Work"
description: "Selected product and engineering work"
showDate: false
showReadingTime: false
showWordCount: false
showAuthor: false
showTableOfContents: false
showPagination: false
---

{{< timeline >}}

{{< timelineItem icon="microsoft" header="Software Engineer - Machine Learning" badge="Mar 2024 - Present" subheader="Microsoft · PowerPoint Copilot" >}}
<ul>
<li>Shipped LLM-powered presentation creation (<strong>700K+ MAU</strong>), translation (<strong>125K+ MAU</strong>), and text-formatting experiences across Web, Win32, and Mac, integrating AI capabilities into production C++, C#, and React codebases</li>
<li>Designed an end-to-end multi-stage LLM workflow for text formatting, covering intent analysis, formatting extraction, and command generation; fine-tuned GPT-4.1 on <strong>10K curated examples</strong> to generate domain-specific editing actions</li>
<li>Improved formatting-agent quality through prompt tuning and research prototypes for single- and multi-agent systems</li>
<li>Designed and shipped the Text-to-Visual experience in PowerPoint Rewrite, enabling one-click text-to-visual conversion; built a Playwright-based evaluation pipeline and LLM-as-judge to score outputs across iterations</li>
<li>Built foundational handoff infrastructure that routes on-canvas Copilot requests into PPT Agent chat, plus a shared licensing layer that standardizes eligibility checks for downstream features</li>
<li>Built a Redis-backed shared undo stack for the PPT headless agent and PowerPoint previewer in M365 Copilot Chat, preserving undo and redo state across agent edits and user changes</li>
<li>Architected cross-platform PPT Agent rollouts independent of a shared release train, cutting the release cycle from <strong>7 days to 2 days</strong></li>
<li>Improved document-grounding upload reliability through end-to-end observability, proactive session recovery, and fixes for retry and concurrency defects, eliminating its largest production failure source</li>
<li>Architected large-file translation for PowerPoint Web using Azure Blob Storage, increasing user success rate by <strong>10%</strong></li>
<li>Prototyped PPT Agent Memory for user context across sessions, winning the Best Project Award at an internal hackathon</li>
</ul>
{{< /timelineItem >}}

{{< timelineItem icon="microsoft" header="Software Engineer" badge="Jul 2023 - Feb 2024" subheader="Microsoft · Stream Playlists" >}}
<ul>
<li>Spearheaded the Add to Playlist plugin for the Microsoft Stream video player with React, enabling seamless playlist curation</li>
<li>Built robust unit and end-to-end test suites (Jest, Playwright), instrumented detailed telemetry, and implemented resilient error handling, driving a measurable <strong>15% increase in monthly active users</strong> for Stream Playlists</li>
<li>Improved accessibility compliance by resolving WCAG issues and fixing key UI bugs to enhance usability and visual polish</li>
</ul>
{{< /timelineItem >}}

{{< timelineItem icon="code" header="Data Scientist Intern" badge="Oct 2022 - Feb 2023" subheader="Murf AI" >}}
<ul>
<li>Pretrained Mixed-Phoneme BERT encoder architecture in PyTorch, focusing on representation learning for text-to-speech</li>
<li>Preprocessed a pre-training dataset (83M sentences) from large unlabeled corpora, applied masking and BPE-based tokenization</li>
<li>Leveraged model and data parallelism to optimize training on multi-GPU infrastructure, enabling efficient large-scale training</li>
<li>Validated architecture changes through BERT pre-training on 1M sentences; delivered scalable, reproducible PyTorch pipelines</li>
</ul>
{{< /timelineItem >}}

{{< timelineItem icon="microsoft" header="Software Engineer Intern" badge="May - Jun 2022" subheader="Microsoft · Viva Learning" >}}
<ul>
<li>Built ASP.NET-based support for SharePoint link-type items in Viva Learning, enabling redirection to external learning content</li>
<li>Optimized ingestion logic by fetching minimal metadata from Graph API, reducing memory usage and improving performance</li>
<li>Built admin diagnostics to show ingestion failures with error details, reducing support load and enabling customer self-triage</li>
</ul>
{{< /timelineItem >}}

{{< /timeline >}}
