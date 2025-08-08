import { Button, FlatList, StyleSheet, Text, TextInput, TouchableOpacity, View } from 'react-native';
import colors from '../constants/colors';
import FontAwesome from '@expo/vector-icons/FontAwesome';
import KeyboardAvoidingViewContainer from '../components/KeyboardAvoidingViewContainer';
import { useCallback, useEffect, useRef, useState } from 'react';
import { makeChatRequest } from '../utils/GPTutils';
import { addUserMessage, getConversation, resetConversation } from '../utils/ConversationHistoryUtils';
import Bubble from '../components/Bubble';
import { HeaderButtons, Item } from 'react-navigation-header-buttons';
import CustomHeaderButton from '../components/CustomHeaderButton';
import MaterialCommunityIcons from '@expo/vector-icons/MaterialCommunityIcons';
import InputContainer from '../components/inputContainer';


export default function ChatScreen(props) { //props - object parameter that contains all properties passed to screen, including navigation - also on a screen 

    const flatlist = useRef();

    //creating a state variable - an array: [value of state variable, method for updating its value]
    const [messageText, setMessageText] = useState("");
    // console.log(messageText); - for watching performance of setMessageText - updates with each keyboard letter input


    //creating a state variable to hold conversation - to be displayed
    const [conversation, setConversation] = useState([]); //its default value is an empty array

    const [loading, setLoading] = useState(false);

    useEffect(() => {
        //creating a button for closing chat
        props.navigation.setOptions({
            headerRight: () => <HeaderButtons HeaderButtonComponent={CustomHeaderButton}>
                <Item
                    title='Clear'
                    iconName='trash-bin-outline'
                    onPress={() => {
                        //want to clear conversation in convo history - after button is clicked
                        setConversation([]);
                        resetConversation();
                    }}
                />
            </HeaderButtons>
        })
    }, []);//useEffect is a component like useState(); and [] is dependency - on which function execution of component depends

    useEffect(() => { // for creating a side effect funvtionality - the one not directly related to the UI output
        resetConversation();
        setConversation([]);
    }, []);

    const sendMessage = useCallback(
        async () => {
            if (messageText === "") return;
            try {
                setLoading(true);
                addUserMessage(messageText);
                setMessageText(""); //setting to empty space new messageText value, so that after clicking send, user can type in sth new and send again
                setConversation([...getConversation()]); //wrapping output of one array into a new array

                await makeChatRequest();
            }
            catch (error) {
                console.log(error);
            }
            finally {
                setConversation([...getConversation()]);//getConversation is from ConversationHistoryUtils.js file
                setLoading(false);
            }
            //console.log(messageText);

        }, [messageText] //every time message changes, recreate funcction with new data;
        //if [] are empty, then never recreating, always using the originally passed in value to the function
    );
    return (
        <KeyboardAvoidingViewContainer>
            <View style={styles.container}>
                <View style={styles.messagesContainer}>
                    {
                        //view shows when there are no msgs yet (conversation length = 0), and when nothing is been loading
                        !loading && conversation.length === 0 &&
                        <View style={styles.emptyContainer}>
                            <MaterialCommunityIcons name="lightbulb-multiple-outline" size={46} color={colors.lightGrey} />
                            <Text style={styles.emptyContainerText}>
                                Type a message to get started
                            </Text>
                        </View>

                    }

                    {
                        conversation.length !== 0 &&

                        <FlatList
                            ref={(ref) => flatlist.current = ref}
                            onLayout={() => flatlist.current.scrollToEnd()}
                            onContentSizeChange={() => flatlist.current.scrollToEnd()} //content = conversation list - when its size changes (new messages are added) scroll to the end
                            style={styles.flatList}
                            data={conversation}
                            renderItem={(itemData) => {
                                const convoItem = itemData.item;
                                const { role, content } = convoItem;
                                if (role === "system") return null;
                                return <Bubble
                                    text={content}
                                    type={role}
                                />

                            }}
                        />

                    }

                    {
                        loading &&
                        <View style={styles.loadingContainer}>
                            <Bubble
                                type="loading"
                            />
                        </View>
                    }

                </View>

                <InputContainer
                    onChangeText={(text) => setMessageText(text)}
                    value={messageText}
                    onPress={sendMessage}
                    placeholder="Type a message to get started!"
                    placeholderTextColor={colors.lightGrey}
                />

            </View>
        </KeyboardAvoidingViewContainer >

    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: colors.greyBg,

    },

    messagesContainer: {
        flex: 1
    },
    flatList: {
        marginHorizontal: 15, //for padding from the sides - left and right,
        paddingVertical: 10 //to have space before convo starts at the top of container
    },
    loadingContainer: {
        position: 'absolute',
        bottom: 0,
        width: '100%',
        alignItems: 'center'
    },
    emptyContainer: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center'
    },
    emptyContainerText: {
        marginTop: 10,
        color: colors.lightGrey,
        fontSize: 18,
        fontFamily: 'regular'
    }
});
