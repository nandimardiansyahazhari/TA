#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <cstdio>
#include <vector>
#include <utility>

using namespace cv;
using namespace std;

int selectedOption = 0; // 0 = none, 1 = fullscreen, 2 = with graph, 3 = settings, 4 = CLI, 5 = shutdown
int rotateMode = 0;     // 0 = none, 90 = 90 deg CW, 270 = 270 deg CW

struct SettingsState {
  bool inSettings;
  bool sshActive;
  bool btActive;
  string wifiSSID;
  string ipAddr;

  int page; // 0 = main settings, 1 = wifi list, 2 = password keyboard, 3 = bluetooth list, 4 = terminal CLI, 10 = wifi connect, 20 = bt connect
  vector<string> wifiList;
  string selectedSSID;
  string typedPassword;

  vector<pair<string, string>> btList; // mac, name
  string selectedBtMac;
  string selectedBtName;

  vector<string> termHistory;
  string typedCommand;
  int kbMode; // 0 = lowercase, 1 = uppercase, 2 = symbols
};

struct Key {
  string label;
  Rect rect;
};

int readRotationConfig() {
  std::ifstream f("rotation.txt");
  if (!f.good()) {
    f.open("../rotation.txt");
    if (!f.good()) {
      f.open("../../rotation.txt");
    }
  }
  int rot = 0;
  if (f.is_open()) {
    f >> rot;
  }
  return rot;
}

void writeRotationConfig(int rot) {
  std::ofstream f("rotation.txt", std::ios::trunc);
  if (f.is_open()) {
    f << rot;
    return;
  }
  f.open("../rotation.txt", std::ios::trunc);
  if (f.is_open()) {
    f << rot;
    return;
  }
  f.open("../../rotation.txt", std::ios::trunc);
  if (f.is_open()) {
    f << rot;
    return;
  }
}

string getIPAddress() {
  FILE* pipe = popen("hostname -I", "r");
  if (!pipe) return "No IP";
  char buffer[128];
  string result = "";
  if (fgets(buffer, sizeof(buffer), pipe) != NULL) {
    result = buffer;
  }
  pclose(pipe);
  if (!result.empty()) {
    size_t end = result.find_first_of(" \n\r");
    if (end != string::npos) {
      result = result.substr(0, end);
    }
  }
  return result.empty() ? "Not Connected" : result;
}

string getWifiSSID() {
  FILE* pipe = popen("iwgetid -r", "r");
  if (!pipe) return "Disconnected";
  char buffer[128];
  string result = "";
  if (fgets(buffer, sizeof(buffer), pipe) != NULL) {
    result = buffer;
  }
  pclose(pipe);
  if (!result.empty() && result.back() == '\n') {
    result.pop_back();
  }
  return result.empty() ? "Disconnected" : result;
}

bool isSSHActive() {
  FILE* pipe = popen("systemctl is-active ssh", "r");
  if (!pipe) return false;
  char buffer[128];
  string result = "";
  if (fgets(buffer, sizeof(buffer), pipe) != NULL) {
    result = buffer;
  }
  pclose(pipe);
  return result.find("active") == 0;
}

bool isBluetoothActive() {
  FILE* pipe = popen("systemctl is-active bluetooth", "r");
  if (!pipe) return false;
  char buffer[128];
  string result = "";
  if (fgets(buffer, sizeof(buffer), pipe) != NULL) {
    result = buffer;
  }
  pclose(pipe);
  return result.find("active") == 0;
}

vector<string> scanWifiNetworks() {
  vector<string> ssids;
  FILE* pipe = popen("nmcli -t -f SSID dev wifi list | sort -u", "r");
  if (!pipe) return ssids;
  char buffer[256];
  while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
    string line = buffer;
    if (!line.empty() && line.back() == '\n') line.pop_back();
    if (!line.empty() && line != "--") {
      ssids.push_back(line);
    }
  }
  pclose(pipe);
  if (ssids.size() > 6) {
    ssids.resize(6);
  }
  return ssids;
}

vector<pair<string, string>> getBluetoothDevices() {
  // Discover new devices
  system("bluetoothctl --timeout 2 scan on > /dev/null 2>&1");
  FILE* pipe = popen("bluetoothctl devices", "r");
  vector<pair<string, string>> list;
  if (!pipe) return list;
  char buffer[256];
  while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
    string line = buffer;
    if (line.find("Device ") == 0) {
      size_t mac_end = line.find(' ', 7);
      if (mac_end != string::npos) {
        string mac = line.substr(7, mac_end - 7);
        string name = line.substr(mac_end + 1);
        if (!name.empty() && name.back() == '\n') name.pop_back();
        list.push_back({mac, name});
      }
    }
  }
  pclose(pipe);
  if (list.size() > 6) {
    list.resize(6);
  }
  return list;
}

vector<string> executeCommand(const string& cmd) {
  vector<string> output;
  string cmd_with_limit = cmd + " 2>&1 | head -n 15";
  FILE* pipe = popen(cmd_with_limit.c_str(), "r");
  if (!pipe) {
    output.push_back("Failed to run command.");
    return output;
  }
  char buffer[256];
  while (fgets(buffer, sizeof(buffer), pipe) != NULL) {
    string line = buffer;
    if (!line.empty() && line.back() == '\n') line.pop_back();
    output.push_back(line);
  }
  pclose(pipe);
  return output;
}

vector<Key> getKeyboardKeys(int mode, bool isTerminal = false, int kbMode = 0) {
  vector<Key> keys;
  vector<vector<string>> layout;
  if (kbMode == 0) { // Lowercase
    layout = {
      {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"},
      {"q", "w", "e", "r", "t", "y", "u", "i", "o", "p"},
      {"a", "s", "d", "f", "g", "h", "j", "k", "l", "-"},
      {"z", "x", "c", "v", "b", "n", "m", "_", ".", "/"}
    };
  } else if (kbMode == 1) { // Uppercase
    layout = {
      {"1", "2", "3", "4", "5", "6", "7", "8", "9", "0"},
      {"Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"},
      {"A", "S", "D", "F", "G", "H", "J", "K", "L", "-"},
      {"Z", "X", "C", "V", "B", "N", "M", "_", ".", "/"}
    };
  } else { // Symbols
    layout = {
      {"~", "!", "@", "#", "$", "%", "^", "&", "*", "("},
      {")", "_", "+", "=", "{", "}", "[", "]", ":", ";"},
      {"\"", "'", "<", ">", "?", "|", "\\", ",", "/", "-"}
    };
  }

  string enterLabel = isTerminal ? "ENTER" : "CONNECT";
  string exitLabel = isTerminal ? "EXIT" : "CANCEL";

  if (mode == 90 || mode == 270) {
    // Landscape Keyboard layout
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 10; ++c) {
        int x = 140 + c * 100;
        int y = 320 + r * 70;
        keys.push_back({layout[r][c], Rect(x, y, 90, 60)});
      }
    }
    keys.push_back({"DEL", Rect(140, 600, 130, 60)});
    keys.push_back({"SHIFT", Rect(285, 600, 130, 60)});
    keys.push_back({"123?", Rect(430, 600, 130, 60)});
    keys.push_back({"SPACE", Rect(575, 600, 200, 60)});
    keys.push_back({enterLabel, Rect(790, 600, 200, 60)});
    keys.push_back({exitLabel, Rect(1005, 600, 130, 60)});
  } else {
    // Portrait Keyboard layout
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 10; ++c) {
        int x = 20 + c * 68;
        int y = 700 + r * 80;
        keys.push_back({layout[r][c], Rect(x, y, 60, 70)});
      }
    }
    keys.push_back({"DEL", Rect(20, 1020, 90, 70)});
    keys.push_back({"SHIFT", Rect(120, 1020, 90, 70)});
    keys.push_back({"123?", Rect(220, 1020, 90, 70)});
    keys.push_back({"SPACE", Rect(320, 1020, 130, 70)});
    keys.push_back({enterLabel, Rect(460, 1020, 130, 70)});
    keys.push_back({exitLabel, Rect(600, 1020, 100, 70)});
  }
  return keys;
}

void onSettingsMouse(int event, int x, int y, int flags, void* userdata) {
  if (event == EVENT_LBUTTONDOWN) {
    SettingsState *state = static_cast<SettingsState*>(userdata);
    int rx = x;
    int ry = y;
    if (rotateMode == 90) {
      rx = y;
      ry = 720 - 1 - x;
    } else if (rotateMode == 270) {
      rx = 1280 - 1 - y;
      ry = x;
    }

    // Check Rotate Button (always present at top right corner)
    bool rotateClicked = false;
    if (rotateMode == 90 || rotateMode == 270) {
      if (rx >= 1140 && rx <= 1240 && ry >= 40 && ry <= 100) {
        rotateClicked = true;
      }
    } else {
      if (x >= 580 && x <= 680 && y >= 40 && y <= 100) {
        rotateClicked = true;
      }
    }

    if (rotateClicked) {
      if (rotateMode == 0) rotateMode = 90;
      else if (rotateMode == 90) rotateMode = 270;
      else rotateMode = 0;
      writeRotationConfig(rotateMode);
      return;
    }

    if (state->page == 0) {
      // Main Settings clicks
      if (rotateMode == 90 || rotateMode == 270) {
        // Landscape clicks
        // SSH Toggle: rx in [680, 1080], ry in [180, 255]
        if (rx >= 680 && rx <= 1080 && ry >= 180 && ry <= 255) {
          if (state->sshActive) {
            system("sudo systemctl stop ssh && sudo systemctl disable ssh");
          } else {
            system("sudo systemctl start ssh && sudo systemctl enable ssh");
          }
          state->sshActive = !state->sshActive;
        }
        // Bluetooth Toggle: rx in [680, 1080], ry in [270, 345]
        else if (rx >= 680 && rx <= 1080 && ry >= 270 && ry <= 345) {
          if (state->btActive) {
            system("sudo systemctl stop bluetooth && sudo systemctl disable bluetooth");
          } else {
            system("sudo systemctl start bluetooth && sudo systemctl enable bluetooth");
          }
          state->btActive = !state->btActive;
        }
        // Rotate: rx in [680, 1080], ry in [360, 435]
        else if (rx >= 680 && rx <= 1080 && ry >= 360 && ry <= 435) {
          if (rotateMode == 0) rotateMode = 90;
          else if (rotateMode == 90) rotateMode = 270;
          else rotateMode = 0;
          writeRotationConfig(rotateMode);
        }
        // Wi-Fi Scan: rx in [100, 500], ry in [360, 435]
        else if (rx >= 100 && rx <= 500 && ry >= 360 && ry <= 435) {
          state->wifiList = scanWifiNetworks();
          state->page = 1;
        }
        // Bluetooth Scan: rx in [100, 500], ry in [455, 530]
        else if (rx >= 100 && rx <= 500 && ry >= 455 && ry <= 530) {
          state->btList = getBluetoothDevices();
          state->page = 3;
        }
        // Kembali: rx in [100, 500], ry in [550, 625]
        else if (rx >= 100 && rx <= 500 && ry >= 550 && ry <= 625) {
          state->inSettings = false;
        }
      } else {
        // Portrait clicks
        // SSH Toggle: x in [60, 660], y in [230, 310]
        if (x >= 60 && x <= 660 && y >= 230 && y <= 310) {
          if (state->sshActive) {
            system("sudo systemctl stop ssh && sudo systemctl disable ssh");
          } else {
            system("sudo systemctl start ssh && sudo systemctl enable ssh");
          }
          state->sshActive = !state->sshActive;
        }
        // Bluetooth Toggle: x in [60, 660], y in [340, 420]
        else if (x >= 60 && x <= 660 && y >= 340 && y <= 420) {
          if (state->btActive) {
            system("sudo systemctl stop bluetooth && sudo systemctl disable bluetooth");
          } else {
            system("sudo systemctl start bluetooth && sudo systemctl enable bluetooth");
          }
          state->btActive = !state->btActive;
        }
        // Rotate: x in [60, 660], y in [450, 530]
        else if (x >= 60 && x <= 660 && y >= 450 && y <= 530) {
          if (rotateMode == 0) rotateMode = 90;
          else if (rotateMode == 90) rotateMode = 270;
          else rotateMode = 0;
          writeRotationConfig(rotateMode);
        }
        // Wi-Fi Scan: x in [60, 660], y in [680, 760]
        else if (x >= 60 && x <= 660 && y >= 680 && y <= 760) {
          state->wifiList = scanWifiNetworks();
          state->page = 1;
        }
        // Bluetooth Scan: x in [60, 660], y in [790, 870]
        else if (x >= 60 && x <= 660 && y >= 790 && y <= 870) {
          state->btList = getBluetoothDevices();
          state->page = 3;
        }
        // Kembali: x in [160, 560], y in [1000, 1100]
        else if (x >= 160 && x <= 560 && y >= 1000 && y <= 1100) {
          state->inSettings = false;
        }
      }
    }
    else if (state->page == 1) {
      // Wi-Fi List Page clicks
      if (rotateMode == 90 || rotateMode == 270) {
        // Landscape clicks
        // Kembali: rx in [100, 400], ry in [550, 630]
        if (rx >= 100 && rx <= 400 && ry >= 550 && ry <= 630) {
          state->page = 0;
          return;
        }
        // SSID 2x3 grid:
        int clickedIdx = -1;
        if (rx >= 100 && rx <= 580) {
          if (ry >= 180 && ry <= 265) clickedIdx = 0;
          else if (ry >= 290 && ry <= 375) clickedIdx = 2;
          else if (ry >= 400 && ry <= 485) clickedIdx = 4;
        } else if (rx >= 680 && rx <= 1160) {
          if (ry >= 180 && ry <= 265) clickedIdx = 1;
          else if (ry >= 290 && ry <= 375) clickedIdx = 3;
          else if (ry >= 400 && ry <= 485) clickedIdx = 5;
        }
        if (clickedIdx >= 0 && clickedIdx < static_cast<int>(state->wifiList.size())) {
          state->selectedSSID = state->wifiList[clickedIdx];
          state->typedPassword = "";
          state->kbMode = 0;
          state->page = 2;
        }
      } else {
        // Portrait clicks
        // Kembali: x in [160, 560], y in [1000, 1100]
        if (x >= 160 && x <= 560 && y >= 1000 && y <= 1100) {
          state->page = 0;
          return;
        }
        // SSID list
        for (int i = 0; i < 6; ++i) {
          if (x >= 60 && x <= 660 && y >= 250 + i * 110 && y <= 250 + i * 110 + 80) {
            if (i < static_cast<int>(state->wifiList.size())) {
              state->selectedSSID = state->wifiList[i];
              state->typedPassword = "";
              state->kbMode = 0;
              state->page = 2;
            }
            break;
          }
        }
      }
    }
    else if (state->page == 2) {
      // Wi-Fi Password Input (Keyboard) clicks
      vector<Key> keys = getKeyboardKeys(rotateMode, false, state->kbMode);
      for (const auto& key : keys) {
        if (key.rect.contains(Point(rx, ry))) {
          if (key.label == "DEL") {
            if (!state->typedPassword.empty()) {
              state->typedPassword.pop_back();
            }
          } else if (key.label == "SHIFT") {
            if (state->kbMode == 1) state->kbMode = 0;
            else state->kbMode = 1;
          } else if (key.label == "123?") {
            if (state->kbMode == 2) state->kbMode = 0;
            else state->kbMode = 2;
          } else if (key.label == "SPACE") {
            state->typedPassword += " ";
          } else if (key.label == "CANCEL") {
            state->page = 1;
          } else if (key.label == "CONNECT") {
            state->page = 10;
          } else {
            state->typedPassword += key.label;
          }
          break;
        }
      }
    }
    else if (state->page == 3) {
      // Bluetooth Device List Page clicks
      if (rotateMode == 90 || rotateMode == 270) {
        // Landscape clicks
        // Kembali: rx in [100, 400], ry in [550, 630]
        if (rx >= 100 && rx <= 400 && ry >= 550 && ry <= 630) {
          state->page = 0;
          return;
        }
        // Device 2x3 grid:
        int clickedIdx = -1;
        if (rx >= 100 && rx <= 580) {
          if (ry >= 180 && ry <= 265) clickedIdx = 0;
          else if (ry >= 290 && ry <= 375) clickedIdx = 2;
          else if (ry >= 400 && ry <= 485) clickedIdx = 4;
        } else if (rx >= 680 && rx <= 1160) {
          if (ry >= 180 && ry <= 265) clickedIdx = 1;
          else if (ry >= 290 && ry <= 375) clickedIdx = 3;
          else if (ry >= 400 && ry <= 485) clickedIdx = 5;
        }
        if (clickedIdx >= 0 && clickedIdx < static_cast<int>(state->btList.size())) {
          state->selectedBtMac = state->btList[clickedIdx].first;
          state->selectedBtName = state->btList[clickedIdx].second;
          state->page = 20;
        }
      } else {
        // Portrait clicks
        // Kembali: x in [160, 560], y in [1000, 1100]
        if (x >= 160 && x <= 560 && y >= 1000 && y <= 1100) {
          state->page = 0;
          return;
        }
        // Device list
        for (int i = 0; i < 6; ++i) {
          if (x >= 60 && x <= 660 && y >= 250 + i * 110 && y <= 250 + i * 110 + 80) {
            if (i < static_cast<int>(state->btList.size())) {
              state->selectedBtMac = state->btList[i].first;
              state->selectedBtName = state->btList[i].second;
              state->page = 20;
            }
            break;
          }
        }
      }
    }
    else if (state->page == 4) {
      // Terminal virtual keyboard clicks
      vector<Key> keys = getKeyboardKeys(rotateMode, true, state->kbMode);
      for (const auto& key : keys) {
        if (key.rect.contains(Point(rx, ry))) {
          if (key.label == "DEL") {
            if (!state->typedCommand.empty()) {
              state->typedCommand.pop_back();
            }
          } else if (key.label == "SHIFT") {
            if (state->kbMode == 1) state->kbMode = 0;
            else state->kbMode = 1;
          } else if (key.label == "123?") {
            if (state->kbMode == 2) state->kbMode = 0;
            else state->kbMode = 2;
          } else if (key.label == "SPACE") {
            state->typedCommand += " ";
          } else if (key.label == "EXIT") {
            state->inSettings = false;
          } else if (key.label == "ENTER") {
            if (!state->typedCommand.empty()) {
              state->termHistory.push_back("ansyah@ansyah:~ $ " + state->typedCommand);
              if (state->typedCommand == "clear") {
                state->termHistory.clear();
              } else if (state->typedCommand == "mulai") {
                state->inSettings = false;
              } else {
                vector<string> out = executeCommand(state->typedCommand);
                for (const auto& line : out) {
                  state->termHistory.push_back(line);
                }
              }
              state->typedCommand = "";
              if (state->termHistory.size() > 50) {
                state->termHistory.erase(state->termHistory.begin(), state->termHistory.begin() + (state->termHistory.size() - 50));
              }
            }
          } else {
            state->typedCommand += key.label;
          }
          break;
        }
      }
    }
  }
}

void onMouse(int event, int x, int y, int flags, void* userdata) {
  if (event == EVENT_LBUTTONDOWN) {
    int rx = x;
    int ry = y;
    if (rotateMode == 90) {
      rx = y;
      ry = 720 - 1 - x;
    } else if (rotateMode == 270) {
      rx = 1280 - 1 - y;
      ry = x;
    }

    // Check Rotate Button (always present at top right corner)
    bool rotateClicked = false;
    if (rotateMode == 90 || rotateMode == 270) {
      if (rx >= 1140 && rx <= 1240 && ry >= 40 && ry <= 100) {
        rotateClicked = true;
      }
    } else {
      if (x >= 580 && x <= 680 && y >= 40 && y <= 100) {
        rotateClicked = true;
      }
    }

    if (rotateClicked) {
      if (rotateMode == 0) rotateMode = 90;
      else if (rotateMode == 90) rotateMode = 270;
      else rotateMode = 0;
      writeRotationConfig(rotateMode);
      return;
    }

    if (rotateMode == 90 || rotateMode == 270) {
      // 2x3 Grid + Centered bottom row in landscape:
      // Button 1 (Fullscreen): rx in [100, 600], ry in [180, 310]
      if (rx >= 100 && rx <= 600 && ry >= 180 && ry <= 310) {
        selectedOption = 1;
      }
      // Button 2 (With Graph): rx in [680, 1180], ry in [180, 310]
      else if (rx >= 680 && rx <= 1180 && ry >= 180 && ry <= 310) {
        selectedOption = 2;
      }
      // Button 3 (Settings): rx in [100, 600], ry in [345, 475]
      else if (rx >= 100 && rx <= 600 && ry >= 345 && ry <= 475) {
        selectedOption = 3;
      }
      // Button 4 (CLI): rx in [680, 1180], ry in [345, 475]
      else if (rx >= 680 && rx <= 1180 && ry >= 345 && ry <= 475) {
        selectedOption = 4;
      }
      // Button 5 (Shutdown): rx in [390, 890], ry in [510, 640]
      else if (rx >= 390 && rx <= 890 && ry >= 510 && ry <= 640) {
        selectedOption = 5;
      }
    } else {
      // Original Portrait Coordinates with 5 buttons:
      // Button 1 (Fullscreen): x in [60, 660], y in [260, 370]
      if (x >= 60 && x <= 660 && y >= 260 && y <= 370) {
        selectedOption = 1;
      }
      // Button 2 (With Graph): x in [60, 660], y in [400, 510]
      else if (x >= 60 && x <= 660 && y >= 400 && y <= 510) {
        selectedOption = 2;
      }
      // Button 3 (Settings): x in [60, 660], y in [540, 650]
      else if (x >= 60 && x <= 660 && y >= 540 && y <= 650) {
        selectedOption = 3;
      }
      // Button 4 (CLI): x in [60, 660], y in [680, 790]
      else if (x >= 60 && x <= 660 && y >= 680 && y <= 790) {
        selectedOption = 4;
      }
      // Button 5 (Shutdown): x in [60, 660], y in [820, 930]
      else if (x >= 60 && x <= 660 && y >= 820 && y <= 930) {
        selectedOption = 5;
      }
    }
  }
}

int main(int argc, char** argv) {
  rotateMode = readRotationConfig();

  namedWindow("Main Menu", WINDOW_NORMAL);
  resizeWindow("Main Menu", 720, 1280);
  moveWindow("Main Menu", 0, 0);
  setMouseCallback("Main Menu", onMouse, nullptr);

  Mat portraitCanvas = Mat::zeros(1280, 720, CV_8UC3);
  Mat landscapeCanvas = Mat::zeros(720, 1280, CV_8UC3);

  while (true) {
    // Sync rotation config in loop
    rotateMode = readRotationConfig();

    if (rotateMode == 90 || rotateMode == 270) {
      landscapeCanvas.setTo(Scalar(20, 20, 20)); // Dark background

      // Draw Title Header (Landscape)
      putText(landscapeCanvas, "COLLISION SYSTEM", Point(450, 75), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 255, 255), 3, LINE_AA);
      putText(landscapeCanvas, "Raspberry Pi 5 Launcher", Point(505, 125), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(180, 180, 180), 2, LINE_AA);
      line(landscapeCanvas, Point(100, 155), Point(1180, 155), Scalar(80, 80, 80), 2);

      // Button 1: Real-time (No Graph / Fullscreen)
      rectangle(landscapeCanvas, Rect(100, 180, 500, 130), Scalar(120, 60, 20), -1); // Teal-blue BGR
      rectangle(landscapeCanvas, Rect(100, 180, 500, 130), Scalar(255, 255, 255), 3);
      putText(landscapeCanvas, "1. DETEKSI FULLSCREEN", Point(195, 255), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

      // Button 2: Real-time (With Graph)
      rectangle(landscapeCanvas, Rect(680, 180, 500, 130), Scalar(20, 100, 20), -1); // Green BGR
      rectangle(landscapeCanvas, Rect(680, 180, 500, 130), Scalar(255, 255, 255), 3);
      putText(landscapeCanvas, "2. DETEKSI DENGAN GRAFIK", Point(765, 255), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

      // Button 3: Settings
      rectangle(landscapeCanvas, Rect(100, 345, 500, 130), Scalar(100, 40, 100), -1); // Purple BGR
      rectangle(landscapeCanvas, Rect(100, 345, 500, 130), Scalar(255, 255, 255), 3);
      putText(landscapeCanvas, "3. PENGATURAN SISTEM", Point(195, 420), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

      // Button 4: Exit to CLI
      rectangle(landscapeCanvas, Rect(680, 345, 500, 130), Scalar(70, 70, 70), -1); // Gray BGR
      rectangle(landscapeCanvas, Rect(680, 345, 500, 130), Scalar(255, 255, 255), 3);
      putText(landscapeCanvas, "4. KELUAR KE CLI", Point(825, 420), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

      // Button 5: Shutdown
      rectangle(landscapeCanvas, Rect(390, 510, 500, 130), Scalar(20, 20, 120), -1); // Red BGR
      rectangle(landscapeCanvas, Rect(390, 510, 500, 130), Scalar(255, 255, 255), 3);
      putText(landscapeCanvas, "5. MATIKAN PI (SHUTDOWN)", Point(475, 585), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

      // ROTATE Button (Gray)
      rectangle(landscapeCanvas, Rect(1140, 40, 100, 60), Scalar(100, 100, 100), -1);
      rectangle(landscapeCanvas, Rect(1140, 40, 100, 60), Scalar(255, 255, 255), 2);
      putText(landscapeCanvas, "ROT", Point(1165, 80), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);

      // Footer info
      putText(landscapeCanvas, "Touch option to select...", Point(520, 675), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(150, 150, 150), 1, LINE_AA);

      Mat rotatedCanvas;
      if (rotateMode == 90) {
        rotate(landscapeCanvas, rotatedCanvas, ROTATE_90_CLOCKWISE);
      } else {
        rotate(landscapeCanvas, rotatedCanvas, ROTATE_90_COUNTERCLOCKWISE);
      }
      imshow("Main Menu", rotatedCanvas);
    } else {
      portraitCanvas.setTo(Scalar(20, 20, 20)); // Dark background

      // Draw Title Header (Portrait)
      putText(portraitCanvas, "COLLISION SYSTEM", Point(130, 120), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(255, 255, 255), 3, LINE_AA);
      putText(portraitCanvas, "Raspberry Pi 5 Launcher", Point(205, 170), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(180, 180, 180), 2, LINE_AA);
      line(portraitCanvas, Point(60, 210), Point(660, 210), Scalar(80, 80, 80), 2);

      // Button 1: Real-time (No Graph / Fullscreen)
      rectangle(portraitCanvas, Rect(60, 260, 600, 110), Scalar(120, 60, 20), -1); // Teal-blue BGR
      rectangle(portraitCanvas, Rect(60, 260, 600, 110), Scalar(255, 255, 255), 3);
      putText(portraitCanvas, "1. DETEKSI FULLSCREEN (NO GRAPH)", Point(100, 325), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

      // Button 2: Real-time (With Graph)
      rectangle(portraitCanvas, Rect(60, 400, 600, 110), Scalar(20, 100, 20), -1); // Green BGR
      rectangle(portraitCanvas, Rect(60, 400, 600, 110), Scalar(255, 255, 255), 3);
      putText(portraitCanvas, "2. DETEKSI DENGAN GRAFIK (NORMAL)", Point(100, 465), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

      // Button 3: Settings
      rectangle(portraitCanvas, Rect(60, 540, 600, 110), Scalar(100, 40, 100), -1); // Purple BGR
      rectangle(portraitCanvas, Rect(60, 540, 600, 110), Scalar(255, 255, 255), 3);
      putText(portraitCanvas, "3. PENGATURAN SISTEM", Point(100, 605), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

      // Button 4: Exit to CLI
      rectangle(portraitCanvas, Rect(60, 680, 600, 110), Scalar(70, 70, 70), -1); // Gray BGR
      rectangle(portraitCanvas, Rect(60, 680, 600, 110), Scalar(255, 255, 255), 3);
      putText(portraitCanvas, "4. KELUAR KE CLI (TERMINAL)", Point(140, 745), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

      // Button 5: Shutdown
      rectangle(portraitCanvas, Rect(60, 820, 600, 110), Scalar(20, 20, 120), -1); // Red BGR
      rectangle(portraitCanvas, Rect(60, 820, 600, 110), Scalar(255, 255, 255), 3);
      putText(portraitCanvas, "5. MATIKAN RASPBERRY PI (SHUTDOWN)", Point(100, 885), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

      // ROTATE Button (Gray)
      rectangle(portraitCanvas, Rect(580, 40, 100, 60), Scalar(100, 100, 100), -1);
      rectangle(portraitCanvas, Rect(580, 40, 100, 60), Scalar(255, 255, 255), 2);
      putText(portraitCanvas, "ROT", Point(605, 80), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);

      // Footer info
      putText(portraitCanvas, "Touch option to select...", Point(240, 1150), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(150, 150, 150), 1, LINE_AA);

      imshow("Main Menu", portraitCanvas);
    }

    int key = waitKey(100);
    if (key == 27) {
      break;
    }

    if (selectedOption != 0) {
      int opt = selectedOption;
      selectedOption = 0; // Reset

      if (opt == 1) {
        destroyAllWindows();
        cout << "Launching Fullscreen Real-time Detection..." << endl;
        string cmd = "./program_optimized --fullscreen --camera 0";
        system(cmd.c_str());
        
        // Re-initialize Main Menu window
        namedWindow("Main Menu", WINDOW_NORMAL);
        resizeWindow("Main Menu", 720, 1280);
        moveWindow("Main Menu", 0, 0);
        setMouseCallback("Main Menu", onMouse, nullptr);
      } 
      else if (opt == 2) {
        destroyAllWindows();
        cout << "Launching Real-time Detection with Graph..." << endl;
        string cmd = "./program_optimized --camera 0";
        system(cmd.c_str());
        
        // Re-initialize Main Menu window
        namedWindow("Main Menu", WINDOW_NORMAL);
        resizeWindow("Main Menu", 720, 1280);
        moveWindow("Main Menu", 0, 0);
        setMouseCallback("Main Menu", onMouse, nullptr);
      } 
      else if (opt == 3) {
        // Run Settings loop
        SettingsState state = { true, false, false, "", "" };
        state.page = 0;
        state.kbMode = 0;
        state.sshActive = isSSHActive();
        state.btActive = isBluetoothActive();
        state.wifiSSID = getWifiSSID();
        state.ipAddr = getIPAddress();

        setMouseCallback("Main Menu", onSettingsMouse, &state);

        while (state.inSettings) {
          rotateMode = readRotationConfig();

          if (rotateMode == 90 || rotateMode == 270) {
            landscapeCanvas.setTo(Scalar(20, 20, 20));

            // ROTATE Button (always top right)
            rectangle(landscapeCanvas, Rect(1140, 40, 100, 60), Scalar(100, 100, 100), -1);
            rectangle(landscapeCanvas, Rect(1140, 40, 100, 60), Scalar(255, 255, 255), 2);
            putText(landscapeCanvas, "ROT", Point(1165, 80), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);

            if (state.page == 0) {
              // Draw Header
              putText(landscapeCanvas, "PENGATURAN SISTEM", Point(450, 75), FONT_HERSHEY_SIMPLEX, 1.1, Scalar(255, 255, 255), 3, LINE_AA);
              line(landscapeCanvas, Point(100, 130), Point(1180, 130), Scalar(80, 80, 80), 2);

              // Left side info: Wi-Fi status
              string ssidText = "Wi-Fi SSID: " + state.wifiSSID;
              string ipText   = "IP Address: " + state.ipAddr;
              putText(landscapeCanvas, ssidText, Point(100, 210), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200, 200, 200), 2, LINE_AA);
              putText(landscapeCanvas, ipText, Point(100, 270), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200, 200, 200), 2, LINE_AA);

              // Wi-Fi Scan & Bluetooth Scan buttons on Left
              rectangle(landscapeCanvas, Rect(100, 360, 400, 75), Scalar(100, 100, 30), -1);
              rectangle(landscapeCanvas, Rect(100, 360, 400, 75), Scalar(255, 255, 255), 2);
              putText(landscapeCanvas, "PILIH WI-FI (SCAN)", Point(180, 405), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);

              rectangle(landscapeCanvas, Rect(100, 455, 400, 75), Scalar(30, 100, 100), -1);
              rectangle(landscapeCanvas, Rect(100, 455, 400, 75), Scalar(255, 255, 255), 2);
              putText(landscapeCanvas, "PILIH BLUETOOTH (SCAN)", Point(160, 500), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);

              // Right side buttons
              // 1. SSH Toggle
              string sshLbl = "SSH: " + string(state.sshActive ? "AKTIF" : "NONAKTIF");
              Scalar sshCol = state.sshActive ? Scalar(20, 150, 20) : Scalar(20, 20, 150);
              rectangle(landscapeCanvas, Rect(680, 180, 400, 75), sshCol, -1);
              rectangle(landscapeCanvas, Rect(680, 180, 400, 75), Scalar(255, 255, 255), 2);
              putText(landscapeCanvas, sshLbl, Point(700, 225), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

              // 2. Bluetooth Toggle
              string btLbl = "BLUETOOTH: " + string(state.btActive ? "AKTIF" : "NONAKTIF");
              Scalar btCol = state.btActive ? Scalar(20, 150, 20) : Scalar(20, 20, 150);
              rectangle(landscapeCanvas, Rect(680, 270, 400, 75), btCol, -1);
              rectangle(landscapeCanvas, Rect(680, 270, 400, 75), Scalar(255, 255, 255), 2);
              putText(landscapeCanvas, btLbl, Point(700, 315), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

              // 3. Screen Rotation Toggle button
              string rotLbl = "ROTASI: " + to_string(rotateMode) + " DERAJAT";
              rectangle(landscapeCanvas, Rect(680, 360, 400, 75), Scalar(120, 60, 20), -1);
              rectangle(landscapeCanvas, Rect(680, 360, 400, 75), Scalar(255, 255, 255), 2);
              putText(landscapeCanvas, rotLbl, Point(700, 405), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

              // KEMBALI Button
              rectangle(landscapeCanvas, Rect(100, 550, 400, 75), Scalar(80, 80, 80), -1);
              rectangle(landscapeCanvas, Rect(100, 550, 400, 75), Scalar(255, 255, 255), 2);
              putText(landscapeCanvas, "KEMBALI KE MENU", Point(200, 595), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);
            }
            else if (state.page == 1) {
              // Wi-Fi List
              putText(landscapeCanvas, "PILIH JARINGAN WI-FI", Point(450, 75), FONT_HERSHEY_SIMPLEX, 1.1, Scalar(255, 255, 255), 3, LINE_AA);
              line(landscapeCanvas, Point(100, 130), Point(1180, 130), Scalar(80, 80, 80), 2);

              if (state.wifiList.empty()) {
                putText(landscapeCanvas, "Tidak ada Wi-Fi ditemukan.", Point(100, 250), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2, LINE_AA);
              } else {
                for (int i = 0; i < static_cast<int>(state.wifiList.size()); ++i) {
                  int col = i % 2;
                  int row = i / 2;
                  int bx = (col == 0) ? 100 : 680;
                  int by = 180 + row * 110;
                  rectangle(landscapeCanvas, Rect(bx, by, 480, 85), Scalar(70, 70, 70), -1);
                  rectangle(landscapeCanvas, Rect(bx, by, 480, 85), Scalar(255, 255, 255), 2);
                  putText(landscapeCanvas, state.wifiList[i], Point(bx + 20, by + 50), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);
                }
              }

              // Kembali button
              rectangle(landscapeCanvas, Rect(100, 550, 300, 80), Scalar(80, 80, 80), -1);
              rectangle(landscapeCanvas, Rect(100, 550, 300, 80), Scalar(255, 255, 255), 2);
              putText(landscapeCanvas, "KEMBALI", Point(190, 600), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);
            }
            else if (state.page == 2) {
              // Wi-Fi Password Input
              putText(landscapeCanvas, "INPUT PASSWORD WI-FI", Point(450, 75), FONT_HERSHEY_SIMPLEX, 1.1, Scalar(255, 255, 255), 3, LINE_AA);
              line(landscapeCanvas, Point(100, 130), Point(1180, 130), Scalar(80, 80, 80), 2);

              string selectedSSIDText = "SSID: " + state.selectedSSID;
              putText(landscapeCanvas, selectedSSIDText, Point(140, 180), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200, 200, 200), 2, LINE_AA);

              // Draw password box
              rectangle(landscapeCanvas, Rect(140, 210, 1000, 60), Scalar(50, 50, 50), -1);
              rectangle(landscapeCanvas, Rect(140, 210, 1000, 60), Scalar(255, 255, 255), 2);
              
              string passDisp = "Password: " + state.typedPassword + "|";
              putText(landscapeCanvas, passDisp, Point(160, 250), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

              // Draw Keyboard keys
              vector<Key> keys = getKeyboardKeys(rotateMode, false, state.kbMode);
              for (const auto& key : keys) {
                Scalar keyCol = Scalar(70, 70, 70);
                if (key.label == "CONNECT") keyCol = Scalar(20, 150, 20);
                else if (key.label == "CANCEL") keyCol = Scalar(20, 20, 150);
                else if (key.label == "DEL") keyCol = Scalar(100, 50, 50);
                
                rectangle(landscapeCanvas, key.rect, keyCol, -1);
                rectangle(landscapeCanvas, key.rect, Scalar(200, 200, 200), 1);
                
                int fontFace = FONT_HERSHEY_SIMPLEX;
                double fontScale = 0.5;
                if (key.label.length() > 3) fontScale = 0.45;
                int thickness = 1;
                int baseline = 0;
                Size textSize = getTextSize(key.label, fontFace, fontScale, thickness, &baseline);
                Point textOrg(key.rect.x + (key.rect.width - textSize.width) / 2,
                              key.rect.y + (key.rect.height + textSize.height) / 2);
                putText(landscapeCanvas, key.label, textOrg, fontFace, fontScale, Scalar(255, 255, 255), thickness, LINE_AA);
              }
            }
            else if (state.page == 3) {
              // Bluetooth Devices List
              putText(landscapeCanvas, "PILIH PERANGKAT BLUETOOTH", Point(450, 75), FONT_HERSHEY_SIMPLEX, 1.1, Scalar(255, 255, 255), 3, LINE_AA);
              line(landscapeCanvas, Point(100, 130), Point(1180, 130), Scalar(80, 80, 80), 2);

              if (state.btList.empty()) {
                putText(landscapeCanvas, "Tidak ada perangkat Bluetooth ditemukan.", Point(100, 250), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2, LINE_AA);
              } else {
                for (int i = 0; i < static_cast<int>(state.btList.size()); ++i) {
                  int col = i % 2;
                  int row = i / 2;
                  int bx = (col == 0) ? 100 : 680;
                  int by = 180 + row * 110;
                  rectangle(landscapeCanvas, Rect(bx, by, 480, 85), Scalar(70, 70, 70), -1);
                  rectangle(landscapeCanvas, Rect(bx, by, 480, 85), Scalar(255, 255, 255), 2);
                  
                  string devName = state.btList[i].second;
                  if (devName.empty()) devName = "Unknown Device";
                  string devMac = state.btList[i].first;
                  
                  putText(landscapeCanvas, devName, Point(bx + 20, by + 40), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255, 255, 255), 2, LINE_AA);
                  putText(landscapeCanvas, devMac, Point(bx + 20, by + 70), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(200, 200, 200), 1, LINE_AA);
                }
              }

              // Kembali button
              rectangle(landscapeCanvas, Rect(100, 550, 300, 80), Scalar(80, 80, 80), -1);
              rectangle(landscapeCanvas, Rect(100, 550, 300, 80), Scalar(255, 255, 255), 2);
              putText(landscapeCanvas, "KEMBALI", Point(190, 600), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);
            }

            else if (state.page == 10) {
              putText(landscapeCanvas, "MENGHUBUNGKAN WI-FI...", Point(450, 250), FONT_HERSHEY_SIMPLEX, 1.1, Scalar(255, 255, 255), 3, LINE_AA);
              string connMsg = "Menghubungkan ke: " + state.selectedSSID;
              putText(landscapeCanvas, connMsg, Point(420, 320), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200, 200, 200), 2, LINE_AA);
            }
            else if (state.page == 20) {
              putText(landscapeCanvas, "MENGHUBUNGKAN BLUETOOTH...", Point(400, 250), FONT_HERSHEY_SIMPLEX, 1.1, Scalar(255, 255, 255), 3, LINE_AA);
              string connMsg = "Menghubungkan ke: " + state.selectedBtName;
              putText(landscapeCanvas, connMsg, Point(420, 320), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200, 200, 200), 2, LINE_AA);
            }

            Mat rotatedCanvas;
            if (rotateMode == 90) {
              rotate(landscapeCanvas, rotatedCanvas, ROTATE_90_CLOCKWISE);
            } else {
              rotate(landscapeCanvas, rotatedCanvas, ROTATE_90_COUNTERCLOCKWISE);
            }
            imshow("Main Menu", rotatedCanvas);
          } else {
            portraitCanvas.setTo(Scalar(20, 20, 20));

            // ROTATE Button (always top right)
            rectangle(portraitCanvas, Rect(580, 40, 100, 60), Scalar(100, 100, 100), -1);
            rectangle(portraitCanvas, Rect(580, 40, 100, 60), Scalar(255, 255, 255), 2);
            putText(portraitCanvas, "ROT", Point(605, 80), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);

            if (state.page == 0) {
              // Draw Header
              putText(portraitCanvas, "PENGATURAN SISTEM", Point(160, 120), FONT_HERSHEY_SIMPLEX, 1.1, Scalar(255, 255, 255), 3, LINE_AA);
              line(portraitCanvas, Point(60, 155), Point(660, 155), Scalar(80, 80, 80), 2);

              // === Info Card: Wi-Fi & IP ===
              rectangle(portraitCanvas, Rect(60, 168, 600, 90), Scalar(35, 35, 55), -1);
              rectangle(portraitCanvas, Rect(60, 168, 600, 90), Scalar(100, 100, 180), 2);
              string ssidText = "Wi-Fi: " + state.wifiSSID;
              string ipText   = "IP  : " + state.ipAddr;
              putText(portraitCanvas, ssidText, Point(80, 200), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(180, 220, 255), 2, LINE_AA);
              putText(portraitCanvas, ipText,   Point(80, 238), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(180, 220, 255), 2, LINE_AA);

              // Buttons start below the card (y=275+)
              // 1. SSH Toggle
              string sshLbl = "SSH: " + string(state.sshActive ? "AKTIF" : "NONAKTIF");
              Scalar sshCol = state.sshActive ? Scalar(20, 150, 20) : Scalar(20, 20, 150);
              rectangle(portraitCanvas, Rect(60, 278, 600, 80), sshCol, -1);
              rectangle(portraitCanvas, Rect(60, 278, 600, 80), Scalar(255, 255, 255), 3);
              putText(portraitCanvas, sshLbl, Point(100, 328), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

              // 2. Bluetooth Toggle
              string btLbl = "BLUETOOTH: " + string(state.btActive ? "AKTIF" : "NONAKTIF");
              Scalar btCol = state.btActive ? Scalar(20, 150, 20) : Scalar(20, 20, 150);
              rectangle(portraitCanvas, Rect(60, 378, 600, 80), btCol, -1);
              rectangle(portraitCanvas, Rect(60, 378, 600, 80), Scalar(255, 255, 255), 3);
              putText(portraitCanvas, btLbl, Point(100, 428), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

              // 3. Screen Rotation Toggle
              string rotLbl = "ROTASI: " + to_string(rotateMode) + " DERAJAT";
              rectangle(portraitCanvas, Rect(60, 478, 600, 80), Scalar(120, 60, 20), -1);
              rectangle(portraitCanvas, Rect(60, 478, 600, 80), Scalar(255, 255, 255), 3);
              putText(portraitCanvas, rotLbl, Point(100, 528), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

              // 4. Scan Wi-Fi Button
              rectangle(portraitCanvas, Rect(60, 680, 600, 80), Scalar(100, 100, 30), -1);
              rectangle(portraitCanvas, Rect(60, 680, 600, 80), Scalar(255, 255, 255), 3);
              putText(portraitCanvas, "PILIH WI-FI (SCAN)", Point(100, 730), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

              // 5. Scan Bluetooth Button
              rectangle(portraitCanvas, Rect(60, 790, 600, 80), Scalar(30, 100, 100), -1);
              rectangle(portraitCanvas, Rect(60, 790, 600, 80), Scalar(255, 255, 255), 3);
              putText(portraitCanvas, "PILIH BLUETOOTH (SCAN)", Point(100, 840), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);

              // KEMBALI Button
              rectangle(portraitCanvas, Rect(160, 1000, 400, 100), Scalar(80, 80, 80), -1);
              rectangle(portraitCanvas, Rect(160, 1000, 400, 100), Scalar(255, 255, 255), 3);
              putText(portraitCanvas, "KEMBALI KE MENU", Point(245, 1060), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);
            }
            else if (state.page == 1) {
              // Wi-Fi List Page
              putText(portraitCanvas, "PILIH JARINGAN WI-FI", Point(160, 120), FONT_HERSHEY_SIMPLEX, 1.1, Scalar(255, 255, 255), 3, LINE_AA);
              line(portraitCanvas, Point(60, 180), Point(660, 180), Scalar(80, 80, 80), 2);

              if (state.wifiList.empty()) {
                putText(portraitCanvas, "Tidak ada Wi-Fi ditemukan.", Point(100, 250), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2, LINE_AA);
              } else {
                for (int i = 0; i < static_cast<int>(state.wifiList.size()); ++i) {
                  int by = 250 + i * 110;
                  rectangle(portraitCanvas, Rect(60, by, 600, 80), Scalar(70, 70, 70), -1);
                  rectangle(portraitCanvas, Rect(60, by, 600, 80), Scalar(255, 255, 255), 2);
                  putText(portraitCanvas, state.wifiList[i], Point(80, by + 50), FONT_HERSHEY_SIMPLEX, 0.65, Scalar(255, 255, 255), 2, LINE_AA);
                }
              }

              // Kembali button
              rectangle(portraitCanvas, Rect(160, 1000, 400, 100), Scalar(80, 80, 80), -1);
              rectangle(portraitCanvas, Rect(160, 1000, 400, 100), Scalar(255, 255, 255), 3);
              putText(portraitCanvas, "KEMBALI", Point(310, 1060), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);
            }
            else if (state.page == 2) {
              // Wi-Fi Password Input (Keyboard)
              putText(portraitCanvas, "INPUT PASSWORD WI-FI", Point(160, 120), FONT_HERSHEY_SIMPLEX, 1.1, Scalar(255, 255, 255), 3, LINE_AA);
              line(portraitCanvas, Point(60, 180), Point(660, 180), Scalar(80, 80, 80), 2);

              string selectedSSIDText = "SSID: " + state.selectedSSID;
              putText(portraitCanvas, selectedSSIDText, Point(60, 230), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(200, 200, 200), 2, LINE_AA);

              // Password box
              rectangle(portraitCanvas, Rect(60, 270, 600, 80), Scalar(50, 50, 50), -1);
              rectangle(portraitCanvas, Rect(60, 270, 600, 80), Scalar(255, 255, 255), 2);
              string passDisp = "Password: " + state.typedPassword + "|";
              putText(portraitCanvas, passDisp, Point(80, 320), FONT_HERSHEY_SIMPLEX, 0.65, Scalar(255, 255, 255), 2, LINE_AA);

              // Draw Keyboard keys
              vector<Key> keys = getKeyboardKeys(rotateMode, false, state.kbMode);
              for (const auto& key : keys) {
                Scalar keyCol = Scalar(70, 70, 70);
                if (key.label == "CONNECT") keyCol = Scalar(20, 150, 20);
                else if (key.label == "CANCEL") keyCol = Scalar(20, 20, 150);
                else if (key.label == "DEL") keyCol = Scalar(100, 50, 50);

                rectangle(portraitCanvas, key.rect, keyCol, -1);
                rectangle(portraitCanvas, key.rect, Scalar(200, 200, 200), 1);

                int fontFace = FONT_HERSHEY_SIMPLEX;
                double fontScale = 0.55;
                if (key.label.length() > 3) fontScale = 0.45;
                int thickness = 2;
                int baseline = 0;
                Size textSize = getTextSize(key.label, fontFace, fontScale, thickness, &baseline);
                Point textOrg(key.rect.x + (key.rect.width - textSize.width) / 2,
                              key.rect.y + (key.rect.height + textSize.height) / 2);
                putText(portraitCanvas, key.label, textOrg, fontFace, fontScale, Scalar(255, 255, 255), thickness, LINE_AA);
              }
            }
            else if (state.page == 3) {
              // Bluetooth Device list
              putText(portraitCanvas, "PILIH PERANGKAT BLUETOOTH", Point(140, 120), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 3, LINE_AA);
              line(portraitCanvas, Point(60, 180), Point(660, 180), Scalar(80, 80, 80), 2);

              if (state.btList.empty()) {
                putText(portraitCanvas, "Tidak ada Bluetooth ditemukan.", Point(100, 250), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2, LINE_AA);
              } else {
                for (int i = 0; i < static_cast<int>(state.btList.size()); ++i) {
                  int by = 250 + i * 110;
                  rectangle(portraitCanvas, Rect(60, by, 600, 80), Scalar(70, 70, 70), -1);
                  rectangle(portraitCanvas, Rect(60, by, 600, 80), Scalar(255, 255, 255), 2);
                  
                  string devName = state.btList[i].second;
                  if (devName.empty()) devName = "Unknown Device";
                  string devMac = state.btList[i].first;
                  
                  putText(portraitCanvas, devName, Point(80, by + 35), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2, LINE_AA);
                  putText(portraitCanvas, devMac, Point(80, by + 65), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1, LINE_AA);
                }
              }

              // Kembali button
              rectangle(portraitCanvas, Rect(160, 1000, 400, 100), Scalar(80, 80, 80), -1);
              rectangle(portraitCanvas, Rect(160, 1000, 400, 100), Scalar(255, 255, 255), 3);
              putText(portraitCanvas, "KEMBALI", Point(310, 1060), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2, LINE_AA);
            }

            else if (state.page == 10) {
              putText(portraitCanvas, "MENGHUBUNGKAN WI-FI...", Point(160, 500), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255, 255, 255), 3, LINE_AA);
              string connMsg = "Menghubungkan ke: " + state.selectedSSID;
              putText(portraitCanvas, connMsg, Point(100, 560), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 200), 2, LINE_AA);
            }
            else if (state.page == 20) {
              putText(portraitCanvas, "MENGHUBUNGKAN BLUETOOTH...", Point(140, 500), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255, 255, 255), 3, LINE_AA);
              string connMsg = "Menghubungkan ke: " + state.selectedBtName;
              putText(portraitCanvas, connMsg, Point(100, 560), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(200, 200, 200), 2, LINE_AA);
            }

            imshow("Main Menu", portraitCanvas);
          }

          int key = waitKey(100);
          if (key == 27) {
            state.inSettings = false;
          }

          // Process background connections dynamically
          if (state.page == 10) {
            string cmd = "nmcli dev wifi connect \"" + state.selectedSSID + "\" password \"" + state.typedPassword + "\"";
            cout << "Executing: " << cmd << endl;
            int ret = system(cmd.c_str());
            
            if (rotateMode == 90 || rotateMode == 270) {
              landscapeCanvas.setTo(Scalar(20, 20, 20));
              if (ret == 0) {
                putText(landscapeCanvas, "KONEKSI BERHASIL!", Point(450, 360), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(20, 255, 20), 3, LINE_AA);
              } else {
                putText(landscapeCanvas, "KONEKSI GAGAL!", Point(480, 360), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(20, 20, 255), 3, LINE_AA);
              }
              Mat rotatedCanvas;
              if (rotateMode == 90) rotate(landscapeCanvas, rotatedCanvas, ROTATE_90_CLOCKWISE);
              else rotate(landscapeCanvas, rotatedCanvas, ROTATE_90_COUNTERCLOCKWISE);
              imshow("Main Menu", rotatedCanvas);
            } else {
              portraitCanvas.setTo(Scalar(20, 20, 20));
              if (ret == 0) {
                putText(portraitCanvas, "KONEKSI BERHASIL!", Point(180, 560), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(20, 255, 20), 3, LINE_AA);
              } else {
                putText(portraitCanvas, "KONEKSI GAGAL!", Point(200, 560), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(20, 20, 255), 3, LINE_AA);
              }
              imshow("Main Menu", portraitCanvas);
            }
            waitKey(2000);

            if (ret == 0) {
              state.wifiSSID = getWifiSSID();
              state.ipAddr = getIPAddress();
              state.page = 0;
            } else {
              state.page = 2;
            }
          }
          else if (state.page == 20) {
            string cmd = "bluetoothctl pair " + state.selectedBtMac + " && bluetoothctl trust " + state.selectedBtMac + " && bluetoothctl connect " + state.selectedBtMac;
            cout << "Executing: " << cmd << endl;
            int ret = system(cmd.c_str());

            if (rotateMode == 90 || rotateMode == 270) {
              landscapeCanvas.setTo(Scalar(20, 20, 20));
              if (ret == 0) {
                putText(landscapeCanvas, "KONEKSI BERHASIL!", Point(450, 360), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(20, 255, 20), 3, LINE_AA);
              } else {
                putText(landscapeCanvas, "KONEKSI GAGAL!", Point(480, 360), FONT_HERSHEY_SIMPLEX, 1.2, Scalar(20, 20, 255), 3, LINE_AA);
              }
              Mat rotatedCanvas;
              if (rotateMode == 90) rotate(landscapeCanvas, rotatedCanvas, ROTATE_90_CLOCKWISE);
              else rotate(landscapeCanvas, rotatedCanvas, ROTATE_90_COUNTERCLOCKWISE);
              imshow("Main Menu", rotatedCanvas);
            } else {
              portraitCanvas.setTo(Scalar(20, 20, 20));
              if (ret == 0) {
                putText(portraitCanvas, "KONEKSI BERHASIL!", Point(180, 560), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(20, 255, 20), 3, LINE_AA);
              } else {
                putText(portraitCanvas, "KONEKSI GAGAL!", Point(200, 560), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(20, 20, 255), 3, LINE_AA);
              }
              imshow("Main Menu", portraitCanvas);
            }
            waitKey(2000);

            if (ret == 0) {
              state.btActive = isBluetoothActive();
              state.page = 0;
            } else {
              state.page = 3;
            }
          }
        }

        // Restore mouse callback to Main Menu when exiting Settings
        setMouseCallback("Main Menu", onMouse, nullptr);
      }
      else if (opt == 4) {
        destroyAllWindows();
        cout << "Exiting to CLI..." << endl;
        system("sudo systemctl stop collision_menu.service");
        exit(0);
      }
      else if (opt == 5) {
        cout << "Shutting down Raspberry Pi..." << endl;
        destroyAllWindows();
        system("sudo poweroff");
        break;
      }
    }
  }

  destroyAllWindows();
  return 0;
}
