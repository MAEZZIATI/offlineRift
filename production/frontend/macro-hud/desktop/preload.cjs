const { contextBridge, ipcRenderer } = require("electron");
function subscribe(channel, callback) {
  const listener = (_event, value) => callback(value);
  ipcRenderer.on(channel, listener);
  if (channel === "rift:frame") ipcRenderer.send("rift:ready");
  return () => ipcRenderer.removeListener(channel, listener);
}
contextBridge.exposeInMainWorld(
  "riftDesktop",
  Object.freeze({
    onFrame: (callback) => subscribe("rift:frame", callback),
    onToggleDetails: (callback) => subscribe("rift:toggle", callback),
    onInteraction: (callback) => subscribe("rift:interaction", callback),
    configure: (context) => ipcRenderer.invoke("rift:configure", context),
    openSettings: () => ipcRenderer.send("rift:settings"),
    setPointerOver: (value) => ipcRenderer.send("rift:pointer", value === true),
  }),
);
