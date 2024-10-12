from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class Assistant:
    def __init__(
        self,
        system_prompt,
        llm,
        message_history=[],
        vector_store=None,
        employee_information=None,
    ):
        self.system_prompt = system_prompt
        self.llm = llm
        self.messages = message_history
        self.vector_store = vector_store
        self.employee_information = employee_information

        self.chain = self._get_conversation_chain()

    def get_response(self, user_input):
        return self.chain.invoke(user_input)

    def _get_conversation_chain(self):
        prompt = ChatPromptTemplate(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder("conversation_history"),
                ("human", "{user_input}"),
            ]
        )

        llm = self.llm

        output_parser = StrOutputParser()

        chain = (
            {
                "retrieved_policy_information": self.vector_store.as_retriever(),
                "employee_information": lambda x: self.employee_information,
                "user_input": RunnablePassthrough(),
                "conversation_history": lambda x: self.messages,
            }
            | prompt
            | llm
            | output_parser
        )
        return chain
