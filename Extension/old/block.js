browser.webRequest.onBeforeRequest.addListener(
    async function (details) {
      const url = new URL(details.url);
      const queryParams = url.searchParams;
  
      // Check for a specific query parameter, e.g., "blockAccess"
      if (queryParams.has("blockAccess")) {
        // You can make an asynchronous request to the server to get the response
        const response = await fetch('http://122.169.31.29');
  
        // Check if the response is -1 and block the request if true
        if (response.status === 200 && (await response.text()) === '-1') {
          return { cancel: true };
        }
      }
  
      return { cancel: false };
    },
    { urls: ["<all_urls>"] },
    ["blocking"]
  );
  