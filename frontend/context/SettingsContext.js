import React from "react";
import { View, Text, Switch, StyleSheet } from "react-native";
import { useTheme } from "../context/ThemeContext";

export default function SettingsScreen() {
    const { darkMode, toggleDarkMode } = useTheme();

    return (
        <View style={styles.container}>
            <Text style={{ color: darkMode ? "white" : "black" }}>
                Dark Mode
            </Text>
            <Switch value={darkMode} onValueChange={toggleDarkMode} />
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        alignItems: "center",
        justifyContent: "center",
    },
});

