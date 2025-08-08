import { Configuration, OpenAI } from "openai";
import { addAssistantMessage, getConversation } from "./ConversationHistoryUtils";

const openai = new OpenAI({
    apiKey: "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"
});

export const makeChatRequest = async () => {
    //console.log(getConversation())
    const response = await openai.chat.completions.create({
        model: "gpt-3.5-turbo",
        messages: getConversation(),

        temperature: 1,
        max_tokens: 256,
        top_p: 1,
        frequency_penalty: 0,
        presence_penalty: 0,
    });

    if (response.choices) {
        let responseText = response.choices[0].message.content;
        //removing line breaks
        responseText = responseText.replace(/(\r\n|\n|\r)/gm, "");
        //not returning responsetext - saved to conversation history
        //console.log(responseText)
        addAssistantMessage(responseText);
        console.log(getConversation()); //messages history - sent by assistant
        return;
    }

    //else:

    throw new Error("The response is in an unsupported format.");
};