import { Image, StyleSheet, Text, View } from "react-native";
import colors from "../constants/colors";
import loadingGif from '../assets/loading.gif';

export default Bubble = (props) => { //props = passed in object
    const { text, type } = props; //take text from props object, and down output the extracted text

    const bubbleStyle = { ...styles.container };
    const wrapperStyle = { ...styles.wrapperStyle };
    const textStyle = { ...styles.textStyle };

    if (type === "assistant") {
        bubbleStyle.backgroundColor = colors.secondary;
        wrapperStyle.justifyContent = "flex-start";
        textStyle.color = colors.textColor;
    }
    return (
        <View style={wrapperStyle}>
            {
                text &&
                <View style={bubbleStyle}>
                    <Text style={textStyle}>{text}</Text>
                </View>
            }
            {
                type === "loading" &&
                <Image
                    source={loadingGif}
                    style={styles.loadingGif}
                />
            }
        </View>
    )
}

const styles = StyleSheet.create({
    container: {
        backgroundColor: colors.primary,
        borderRadius: 25, //roundness off the boxes/bubbles around text in msgs; increase to be more round - but increase than horizontalPadding as well
        padding: 10,
        paddingHorizontal: 12, //between the messages
        marginBottom: 10,
        maxWidth: "90%"

    },
    wrapperStyle: {
        flexDirection: "row",
        justifyContent: "flex-end",
    },
    textStyle: {
        color: 'white',
        fontFamily: 'regular'
    },
    loadingGif: {
        height: 100,

    }
})