import { StyleSheet, TextInput, TouchableOpacity, View } from "react-native";
import colors from "../constants/colors";
import FontAwesome from '@expo/vector-icons/FontAwesome';

export default inputContainer = (props) => {
    //extracting values from props:
    const { onChangeText, value, onPress, placeholder, placeholderTextColor } = props;

    return (
        <View style={styles.container}>
            <TextInput
                style={styles.textBox}
                placeholder={placeholder}
                onChangeText={onChangeText}
                value={value}
                placeholderTextColor={placeholderTextColor}
            />

            <TouchableOpacity style={styles.sendButton} onPress={onPress}>
                <FontAwesome name="send-o" size={18} color="white" />
            </TouchableOpacity>

        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flexDirection: 'row',
        backgroundColor: 'white',
        padding: 10
    },
    textBox: {
        flex: 1,
        fontFamily: 'regular',
        letterSpacing: 0.5 //or don't use
    },
    sendButton: {
        backgroundColor: colors.primary,
        width: 35,
        height: 35,
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 50
    },
});