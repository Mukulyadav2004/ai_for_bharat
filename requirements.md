# Requirements Document: Insight Agents System

## Introduction

The Insight Agents (IA) system is an LLM-based conversational multi-agent platform designed to help e-commerce sellers convert complex business data into actionable insights through automated information retrieval. The system addresses two key seller challenges:

1. Difficulty discovering and effectively utilizing available programs and tools
2. Struggling to understand and utilize rich data from various sources

IA serves as a force multiplier for sellers, reducing cognitive load and increasing the speed at which sellers make good business decisions. The system is built on a plan-and-execute paradigm with a hierarchical manager-worker multi-agent architecture, designed for comprehensive coverage, high accuracy, and low latency.

The system supports two main types of requests:

- **Descriptive Analytics**: Presents data according to specified queries (e.g., "What were my sales and traffic for the top 10 products last month?", "How does my monthly sales change year over year?")
- **Diagnostic Analysis**: Provides summarization, benchmarking, and analytical insights (e.g., "How does my business perform?", "How is my business doing with respect to my benchmarks?")

The system leverages Retrieval-Augmented Generation (RAG) with tabular data through a robust API-based data model, achieving 89.5% question-level accuracy with sub-15-second P90 latency. The architecture uses specialized lightweight models (autoencoder for OOD, BERT for routing) to optimize latency while maintaining high accuracy.

## Glossary

- **Manager_Agent**: Control layer agent responsible for OOD detection, agent routing, and query augmentation
- **Worker_Agent**: Specialized agents that execute specific resolution paths (Data Presenter or Insight Generator)
- **Data_Presenter_Agent**: Worker agent that retrieves and presents data through customized data retrieval and aggregation
- **Insight_Generator_Agent**: Worker agent that performs domain-aware diagnostic analysis with dynamically injected domain knowledge
- **OOD_Detection**: Autoencoder-based out-of-domain detection using reconstruction error threshold (μ_id + λ * σ_id)
- **Agent_Router**: Fine-tuned BERT-based classifier (33M parameters) that categorizes queries between resolution paths
- **Query_Augmenter**: Component that clarifies, rewrites, and expands queries to reduce ambiguity (especially time ranges)
- **Data_Workflow_Planner**: LLM-based component performing task decomposition, API/function selection, and payload generation with slot filling
- **Data_Workflow_Executor**: Component executing data retrieval, aggregation, transformation, and post-processing operations
- **RAG**: Retrieval-Augmented Generation framework leveraging seller data in tabular format through API-based retrieval
- **Guardrails**: Post-processing safety mechanisms preventing PII leakage, toxic messages, and policy violations
- **Seller**: E-commerce business owner or operator using the system
- **In_Domain**: Queries where at least part of the question can be answered based on available data
- **Domain_Aware_Routing**: Few-shot learning based LLM classifier for domain-specific branch routing in insight generator
- **ICL**: In-Context Learning using few-shot examples and chain-of-thought prompting for generation
- **Schema_Linking**: Process of aligning entity references in queries to schema tables or columns
- **Slot_Filling**: Populating required input parameters for chosen APIs or functions
- **CoT**: Chain-of-Thought reasoning for comprehensive instruction adherence

## Requirements

### Requirement 1: Autoencoder-Based Out-of-Domain Detection

**User Story:** As a seller, I want the system to quickly and accurately recognize when my question is outside its capabilities, so that I receive honest feedback rather than incorrect or fabricated answers.

#### Acceptance Criteria

1. WHEN a user submits a query, THE OOD_Detection SHALL encode the query using a sentence transformer and compute reconstruction error using a trained autoencoder
2. THE Autoencoder SHALL have one hidden layer with dimension 64 and be trained on in-domain questions to minimize reconstruction error
3. WHEN the reconstruction error exceeds the threshold (μ_id + λ * σ_id where λ=4), THE OOD_Detection SHALL classify the query as out-of-domain
4. IF a query is detected as out-of-domain, THEN THE Manager_Agent SHALL reject the query and provide a clear explanation of system capabilities
5. WHEN an out-of-domain query is detected, THE System SHALL NOT route the query to any Worker_Agent
6. THE OOD_Detection SHALL complete evaluation in less than 0.01 seconds per query
7. THE OOD_Detection SHALL favor precision over recall to act as an initial screening layer
8. THE OOD_Detection SHALL execute in parallel with Agent_Router to minimize overall latency

### Requirement 2: BERT-Based Agent Routing

**User Story:** As a seller, I want my questions automatically directed to the right analysis capability with low latency, so that I get the most relevant type of response quickly.

#### Acceptance Criteria

1. WHEN a user submits an in-domain query, THE Agent_Router SHALL classify it as either data presenter or insight generator using a fine-tuned BERT model (bge-small-en-v1.5, 33M parameters)
2. THE Agent_Router SHALL be trained on super-sampled data with 300 data presenter questions and 300 insight generator questions
3. WHEN a query requests descriptive analytics or data presentation, THE Agent_Router SHALL route to the Data_Presenter_Agent
4. WHEN a query requests diagnostic analysis, summarization, or benchmarking, THE Agent_Router SHALL route to the Insight_Generator_Agent
5. THE Agent_Router SHALL complete classification in approximately 0.3 seconds per query
6. THE Agent_Router SHALL achieve at least 80% classification accuracy
7. THE Agent_Router SHALL execute in parallel with OOD_Detection to minimize overall latency
8. WHEN both OOD_Detection and Agent_Router complete, THE Manager_Agent SHALL proceed only if query is in-domain

### Requirement 3: Query Augmentation for Ambiguity Reduction

**User Story:** As a seller, I want to ask questions with relative time references like "last week", so that I don't need to calculate exact dates myself.

#### Acceptance Criteria

1. WHEN a user submits a query with ambiguous time references, THE Query_Augmenter SHALL inject contextual information including today's date, current week start/end dates, and calendar boundaries
2. THE Query_Augmenter SHALL clarify, rewrite, and expand queries to reduce ambiguity
3. THE Query_Augmenter SHALL provide specific instructions for time range interpretation (e.g., "week" refers to calendar week, not rolling 7 days)
4. WHEN query augmentation is complete, THE Manager_Agent SHALL pass the augmented query to the selected Worker_Agent
5. THE Query_Augmenter SHALL preserve the original user intent while adding clarifying details
6. THE Query_Augmenter SHALL augment the prompt for subsequent LLM calls in worker agents

### Requirement 4: API-Based Data Workflow Planning

**User Story:** As a seller, I want the system to accurately retrieve data from available APIs, so that I can trust the information presented to me.

#### Acceptance Criteria

1. WHEN the Data_Presenter_Agent receives a routed query, THE Data_Workflow_Planner SHALL decompose the query into executable steps based on available data API specifications
2. THE Data_Workflow_Planner SHALL use chain-of-thought reasoning and few-shot learning to ensure comprehensive instruction adherence
3. WHEN decomposing queries, THE Data_Workflow_Planner SHALL perform task decomposition and planning in a divide-and-conquer manner
4. WHEN tasks are identified, THE Data_Workflow_Planner SHALL select appropriate APIs/functions by matching query requirements to API names, descriptions, and column metadata
5. THE Data_Workflow_Planner SHALL perform schema linking to align entity references in queries to schema tables or columns
6. WHEN API selection is complete, THE Data_Workflow_Planner SHALL generate payloads with slot filling for required input parameters
7. THE Data_Workflow_Planner SHALL perform secondary out-of-scope detection by comparing queries against dataset metadata
8. WHEN queries exceed available data boundaries, THE Data_Workflow_Planner SHALL explicitly provide an "out" option to prevent hallucination
9. THE Data_Workflow_Planner SHALL leverage tool metadata and few-shot examples stored in memory

### Requirement 5: Data Workflow Execution and Post-Processing

**User Story:** As a seller, I want retrieved data to be properly formatted and aggregated, so that I can easily understand the information.

#### Acceptance Criteria

1. WHEN the Data_Workflow_Planner completes planning, THE Data_Workflow_Executor SHALL execute data retrieval operations by calling selected APIs with generated payloads
2. THE Data_Workflow_Executor SHALL perform data aggregation and transformation operations using external calculation tools to avoid LLM calculation errors
3. WHEN data retrieval is complete, THE Data_Workflow_Executor SHALL apply post-processing including data reformatting, column renaming, and semantic matching-based column filtering
4. THE Data_Workflow_Executor SHALL handle API errors gracefully and provide meaningful error messages
5. WHEN processing is complete, THE Data_Presenter_Agent SHALL generate natural language responses using in-context learning with few-shot examples to guide the desired format

### Requirement 6: Domain-Aware Insight Generation

**User Story:** As a seller, I want diagnostic insights that leverage domain-specific knowledge, so that I receive relevant and actionable recommendations.

#### Acceptance Criteria

1. WHEN the Insight_Generator_Agent receives a routed query, THE Data_Workflow_Executor SHALL decompose the query into domain-specific categories (performance, benchmarking, recommendation, etc.)
2. THE Insight_Generator_Agent SHALL use a few-shot learning based LLM classifier for domain-specific branch routing
3. WHEN domain categories are identified, THE Insight_Generator_Agent SHALL select predefined domain-aware resolution paths
4. THE Domain_Aware_Resolution_Path SHALL include associated analytical analysis (data aggregation, time series seasonal & trend analysis, benchmark analysis)
5. THE Domain_Aware_Resolution_Path SHALL include domain-aware knowledge, prompt templates, and few-shot examples for final insight generation
6. WHEN data is retrieved, THE Insight_Generator_Agent SHALL dynamically inject domain knowledge based on the query
7. THE Insight_Generator_Agent SHALL generate insights using in-context learning with chain-of-thought reasoning and domain-specific few-shot examples provided by domain experts

### Requirement 7: Parallel Execution with Early Termination

**User Story:** As a seller, I want fast responses to my questions, so that I can make timely business decisions.

#### Acceptance Criteria

1. WHEN a query is received, THE Manager_Agent SHALL execute OOD_Detection and Agent_Router in parallel
2. THE System SHALL initiate both Data_Presenter_Agent and Insight_Generator_Agent branches concurrently
3. WHEN the Agent_Router determines the correct branch, THE System SHALL terminate the incorrect branch early
4. THE System SHALL trade off increased infrastructure and computation costs for latency reduction
5. THE System SHALL achieve P90 latency of less than 15 seconds for end-to-end query processing

### Requirement 8: Response Safety with Guardrails

**User Story:** As a platform operator, I want all system responses to comply with safety policies, so that sellers receive appropriate and compliant information.

#### Acceptance Criteria

1. WHEN a Worker_Agent generates a response, THE Manager_Agent SHALL apply guardrails as a post-processing step before returning results to the user
2. THE Guardrails SHALL evaluate responses for PII leakage, toxic messages, and policy violations
3. IF a response contains PII, toxic content, or policy violations, THEN THE Manager_Agent SHALL modify or reject the response
4. WHEN guardrails are satisfied, THE Manager_Agent SHALL return the approved response to the user

### Requirement 9: Comprehensive Analytical Reasoning

**User Story:** As a seller, I want diagnostic insights that explain why things are happening in my business, so that I can understand root causes and take appropriate action.

#### Acceptance Criteria

1. WHEN a diagnostic query is processed, THE Insight_Generator_Agent SHALL retrieve data across multiple relevant business dimensions
2. THE Insight_Generator_Agent SHALL apply analytical techniques including time series analysis, seasonal & trend analysis, and benchmark analysis
3. WHEN analyzing data, THE Insight_Generator_Agent SHALL identify correlations, causal factors, and patterns
4. THE Insight_Generator_Agent SHALL provide both explanatory insights (why something happened) and actionable recommendations (what to do next)
5. THE Insight_Generator_Agent SHALL use domain expert knowledge to enhance insight quality

### Requirement 10: High Accuracy Data Retrieval

**User Story:** As a seller, I want data retrieval to be more accurate than text-to-SQL approaches, so that I can trust the system's outputs.

#### Acceptance Criteria

1. THE Data_Workflow_Planner SHALL use API-based retrieval rather than text-to-SQL to avoid syntax errors and hallucinations
2. THE System SHALL leverage company's internal data APIs with natural structure and constraints
3. THE System SHALL use external calculation tools for data transformation to avoid LLM calculation errors
4. THE System SHALL achieve at least 89.5% question-level accuracy on benchmarking datasets
5. Question-level accuracy SHALL be defined as the percentage of questions with correctness, completeness, and relevancy scores above 0.8

### Requirement 11: Response Quality Evaluation

**User Story:** As a system operator, I want to measure response quality across multiple dimensions, so that I can monitor and improve system performance.

#### Acceptance Criteria

1. THE System SHALL evaluate responses for relevance, defined as the ratio of key words from the question addressed in the response to total key words in the question
2. THE System SHALL evaluate responses for correctness, defined as the ratio of correct insights in the response to total insights in the response
3. THE System SHALL evaluate responses for completeness, defined as the ratio of required insights in the response to total required insights
4. WHEN all three metrics (relevance, correctness, completeness) exceed 0.8, THE System SHALL consider the question answered accurately
5. THE System SHALL track question-level accuracy as the primary performance metric

### Requirement 12: Tool Learning and Usage

**User Story:** As a system designer, I want the system to effectively select and use available tools and APIs, so that data retrieval is comprehensive and accurate.

#### Acceptance Criteria

1. THE Data_Workflow_Planner SHALL store tool metadata including API/function names, descriptions, and parameter schemas
2. THE Data_Workflow_Planner SHALL store few-shot examples demonstrating tool usage patterns
3. WHEN selecting tools, THE Data_Workflow_Planner SHALL match query requirements to tool capabilities using LLM-based tool selection
4. THE System SHALL provide detailed API/function descriptions to the LLM to enable customized data retrieval and aggregation
5. THE System SHALL support both data retrieval APIs and calculation/transformation functions
