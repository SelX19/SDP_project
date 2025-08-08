import { KeyboardAvoidingView, Platform } from "react-native"

export default KeyboardAvoidingViewContainer = (props) => {

    /* in case that some Android devices cannot work with defined KeyBoardAvoidingView:

    if (Platform.OS === 'android') {
        return props.children;
    }
        */

    return <KeyboardAvoidingView
        style={{ flex: 1 }} behavior="padding" keyboardVerticalOffset={100}
    >
        {props.children}
    </KeyboardAvoidingView>
}