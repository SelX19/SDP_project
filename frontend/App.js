import "react-native-url-polyfill/auto";
import { StatusBar } from 'expo-status-bar';
import { StyleSheet, View } from 'react-native';
import { NavigationContainer, DefaultTheme, DarkTheme } from '@react-navigation/native';
import MainNavigator from './components/MainNavigator';
import { useFonts } from 'expo-font';
import * as SplashScreen from 'expo-splash-screen';
import { useCallback } from "react";
import { HeaderButtonsProvider } from "react-navigation-header-buttons/HeaderButtonsProvider";
import { SafeAreaProvider } from "react-native-safe-area-context";
import { SettingsProvider } from "./context/SettingsContext";
import { ThemeProvider, useTheme } from "./context/ThemeContext";

SplashScreen.preventAutoHideAsync();

export default function App() {
  const [fontsLoaded] = useFonts({
    "regular": require("./assets/fonts/Poppins-Regular.ttf")
  });

  const onLayoutRootView = useCallback(async () => {
    if (fontsLoaded) {
      await SplashScreen.hideAsync();
    }
  }, [fontsLoaded]);

  if (!fontsLoaded) {
    return null;
  }

  return (
    <SafeAreaProvider>
      <View style={{ flex: 1 }} onLayout={onLayoutRootView}>
        <ThemeProvider>
          <AppWithTheme />
        </ThemeProvider>
      </View>
    </SafeAreaProvider>
  );
}

function AppWithTheme() {
  const { darkMode } = useTheme();

  return (
    <View style={{ flex: 1 }}>
      <NavigationContainer theme={darkMode ? DarkTheme : DefaultTheme}>
        <HeaderButtonsProvider stackType="js">
          <MainNavigator />
        </HeaderButtonsProvider>
        <StatusBar style={darkMode ? "light" : "dark"} />
      </NavigationContainer>

      {/* Dark overlay with opacity */}
      {darkMode && (
        <View
          style={{
            ...StyleSheet.absoluteFillObject, // full screen overlay
            backgroundColor: "rgba(0,0,0,0.3)", // adjust 0.3 → 0.6 for stronger dim
          }}
          pointerEvents="none" // ensures overlay doesn’t block touches
        />
      )}
    </View>
  );
}


const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});

