import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import ChatScreen from '../screens/ChatScreen';
import ImageScreen from '../screens/ImageScreen';
import SettingsScreen from '../screens/SettingsScreen';
import Entypo from '@expo/vector-icons/Entypo';
import MaterialIcons from '@expo/vector-icons/MaterialIcons';
import SimpleLineIcons from '@expo/vector-icons/SimpleLineIcons';
import colors from '../constants/colors';

const options = {
    headerTitleStyle: { //built-in ReactNative options
        fontFamily: 'regular',
        colors: colors.textColor
    },
    tabBarLabelStyle: { //built-in ReactNative options
        fontFamily: 'regular',
        colors: colors.textColor
    },
    tabBarShowLabel: false //built-in ReactNative options - by using this one, we hide the text below the icons in the lower nav bar (tab bar)
}

const Tab = createBottomTabNavigator();

const MainNavigator = () => {
    return (
        <Tab.Navigator>
            <Tab.Screen name="Chat" component={ChatScreen} options={
                {
                    ...options,
                    headerTitleAlign: 'center',
                    tabBarIcon: ({ color, size }) => {
                        return <Entypo name="chat" size={size} color={color} />
                    }
                }
            } />
            <Tab.Screen name="Images" component={ImageScreen} options={
                {
                    ...options,
                    headerTitleAlign: 'center',
                    tabBarIcon: ({ color, size }) => {
                        return <MaterialIcons name="image" size={size} color={color} />
                    }
                }
            } />
            <Tab.Screen name="Settings" component={SettingsScreen} options={
                {
                    ...options,
                    headerTitleAlign: 'center',
                    tabBarIcon: ({ color, size }) => {
                        return <SimpleLineIcons name="settings" size={size} color={color} />
                    }
                }
            } />
        </Tab.Navigator>

    );
};

export default MainNavigator;