// export let conversation = [];  OR

let conversation = []

export const getConversation = () => {
    return conversation;
} // or => conversation;

export const initConversation = () => {
    addSystemMessage("Your name is a TouristBot"); //instruction from the system to the assistant
}

export const addUserMessage = (messageText) => {
    conversation.push({
        role: "user",
        content: messageText
    })
}

//3 roles in chatGPT model

export const addAssistantMessage = (messageText) => {
    conversation.push({
        role: "assistant",
        content: messageText
    })
}

export const addSystemMessage = (messageText) => {
    conversation.push({
        role: "system",
        content: messageText
    })
}

export const resetConversation = () => {
    conversation = []; //clear conversation
    initConversation(); //and then reinitialize it

}