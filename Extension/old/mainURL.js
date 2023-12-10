// background.js
function logNewTabUrl(tabId, changeInfo, tab) {
	// Check if the tab has finished loading
	if (changeInfo.status === 'complete') {
	  console.log("Newly loaded tab URL:", tab.url);
	}
  }
  
  // Add an event listener for tab updates
  browser.tabs.onUpdated.addListener(logNewTabUrl);
  