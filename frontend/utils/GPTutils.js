import { Platform } from 'react-native';
import { addAssistantMessage, getConversation } from "./ConversationHistoryUtils";

export const makeChatRequest = async (message) => {
    // Set backend URL depending on platform
    let backendUrl = "";

    if (Platform.OS === "android") {
        backendUrl = "http://10.0.2.2:80/chat";  // works for emulator
    } else {
        backendUrl = "http://192.168.0.24:80/chat";  // use LAN IP for iOS simulator + real device
    }

    //or: const backendUrl = "https://80b3348d4f71.ngrok-free.app/chat";

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
