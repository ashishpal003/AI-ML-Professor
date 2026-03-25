from rag_src.generation.llm import LLMService
from rag_src.utils.logger import get_logger
from rag_src.utils.exceptions import MyException
from langchain_core.messages import HumanMessage

logger = get_logger(__name__)

class QueryRewriter:

    def __init__(self, model_name):
        self.llm = LLMService(model_name=model_name)

    async def rewrite(self, query: str) -> str:
        prompt = f"""
You are an expert tutor AI assistant designed to help students learn Machine Learning and AI.
Your task is to rewrite user questions to be more specific, educational, and effective for retrieving information from academic textbooks.

### Guidelines
1.  **Expand Abbrevations:** Expand AI/ML acronyms (e.g., "NN" -> "Neural Networks").
2.  **Add Educational Context:** Focus on defining, explaining, or providing examples.
3.  **Specify Scope:** If a query is too broad, add context relevant to learning.
4.  **Keep it Focused:** Do not answer the question; only rewrite it.

Note: Please dont not add any thing extra in the response just provide the **Rewritten Query**

### Examples

**User Query:** what is overfit
**Rewritten Query:** Define overfitting in machine learning, explain its causes, and discuss common regularization techniques to prevent it.

**User Query:** bias variance trade off
**Rewritten Query:** Explain the bias-variance tradeoff in supervised learning and how it affects model generalization.

**User Query:** difference between cnn and rnn
**Rewritten Query:** Compare and contrast Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) in terms of architecture and typical use cases.

**User Query:** SGD
**Rewritten Query:** Describe Stochastic Gradient Descent (SGD) algorithm, its advantages over batch gradient descent, and its role in training deep learning models.

**User Query:** why model not working
**Rewritten Query:** What are the common reasons a machine learning model fails to learn, including data issues, hyperparameter errors, and overfitting?

### New Request
**User Query:** {query}
**Rewritten Query:**
"""
        try:
            rewritten = await self.llm.agenerate([HumanMessage(content=prompt)])
            logger.info(f"Rewritten query: {rewritten}")
            return rewritten.strip()

        except Exception as e:
            logger.warning(f"Rewrite failed: {e}")
            return query #fallback