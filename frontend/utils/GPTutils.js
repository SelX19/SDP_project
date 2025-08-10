import { Platform } from 'react-native';
import { addAssistantMessage, getConversation } from "./ConversationHistoryUtils";

export const makeChatRequest = async (message) => {
    // Set backend URL depending on platform
    let backendUrl = "";

    if (Platform.OS === "android") {
        backendUrl = "http://10.0.2.2:80/chat";
    } else if (Platform.OS === "ios") {
        backendUrl = "http://127.0.0.1:80/chat";
    } else {
        backendUrl = "http://192.168.0.24:80/chat";  // replace with your PC's local IP
    }

    const response = await fetch(backendUrl, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ message }),
    });

    const data = await response.json();

    addAssistantMessage(data.answer);
};
