import { Button, FlatList, StyleSheet, Text, TextInput, TouchableOpacity, View, ImageBackground } from 'react-native';
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
import { BlurView } from 'expo-blur';


export default function ChatScreen(props) {

    const flatlist = useRef();
    const [messageText, setMessageText] = useState("");
    const [conversation, setConversation] = useState([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        props.navigation.setOptions({
            headerRight: () => <HeaderButtons HeaderButtonComponent={CustomHeaderButton}>
                <Item
                    title='Clear'
                    iconName='trash-bin-outline'
                    onPress={() => {
                        setConversation([]);
                        resetConversation();
                    }}
                />
            </HeaderButtons>
        })
    }, []);

    useEffect(() => {
        resetConversation();
        setConversation([]);
    }, []);

    const sendMessage = useCallback(
        async () => {
            if (messageText === "") return;
            try {
                setLoading(true);
                addUserMessage(messageText);
                setMessageText("");
                setConversation([...getConversation()]);
                await makeChatRequest(messageText);
            }
            catch (error) {
                console.log(error);
            }
            finally {
                setConversation([...getConversation()]);
                setLoading(false);
            }
        }, [messageText]
    );

    return (
        <KeyboardAvoidingViewContainer>
            <ImageBackground
                source={require('../assets/mostar2.jpeg')}
                style={styles.background}
                resizeMode="cover"
            >
                {/* Overlay to control opacity */}
                {/* Blur overlay instead of dark opacity */}
                <BlurView intensity={50} tint="dark" style={StyleSheet.absoluteFillObject} />


                <View style={styles.container}>
                    <View style={styles.messagesContainer}>
                        {!loading && conversation.length === 0 &&
                            <View style={styles.emptyContainer}>
                                <Text style={styles.emptyContainerText}>
                                    Welcome back!
                                </Text>
                                <MaterialCommunityIcons name="lightbulb-multiple-outline" size={46} color={colors.greyBg} />
                            </View>
                        }

                        {conversation.length !== 0 &&
                            <FlatList
                                ref={(ref) => flatlist.current = ref}
                                onLayout={() => flatlist.current.scrollToEnd()}
                                onContentSizeChange={() => flatlist.current.scrollToEnd()}
                                style={styles.flatList}
                                data={conversation}
                                renderItem={(itemData) => {
                                    const convoItem = itemData.item;
                                    const { role, content } = convoItem;
                                    if (role === "system") return null;
                                    return <Bubble text={content} type={role} />
                                }}
                            />
                        }

                        {loading &&
                            <View style={styles.loadingContainer}>
                                <Bubble type="loading" />
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
            </ImageBackground>
        </KeyboardAvoidingViewContainer>
    );
}

const styles = StyleSheet.create({
    background: {
        flex: 1,
    },
    container: {
        flex: 1,
    },
    messagesContainer: {
        flex: 1
    },
    flatList: {
        marginHorizontal: 15,
        paddingVertical: 10
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
        marginTop: 2,
        marginBottom: 10,
        color: colors.greyBg,
        fontSize: 22,
        fontFamily: 'regular'
    }
});



