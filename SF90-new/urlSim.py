# import re

# def extract_domain_parts(url):
#     match = re.search(r"https?://(?:www\.)?([^/]+)", url)
#     if match:
#         domain = match.group(1)
#         parts = domain.split('.')
#         return parts
#     return None

# def calculate_string_similarity(str1, str2):
#     m = len(str1)
#     n = len(str2)
#     dp = [[0] * (n + 1) for _ in range(m + 1)]

#     for i in range(m + 1):
#         for j in range(n + 1):
#             if i == 0:
#                 dp[i][j] = j
#             elif j == 0:
#                 dp[i][j] = i
#             elif str1[i - 1] == str2[j - 1]:
#                 dp[i][j] = dp[i - 1][j - 1]
#             else:
#                 dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

#     max_len = max(len(str1), len(str2))
#     similarity = 1 - (dp[m][n] / max_len)
#     return similarity

# def get_tld_similarity_score(url1, url2):
#     parts1 = extract_domain_parts(url1)
#     parts2 = extract_domain_parts(url2)
#     print(parts1[-1],parts2[-1],"tld")
#     if parts1 is not None and parts2 is not None:
#         tld_similarity = calculate_string_similarity(parts1[-1], parts2[-1])
#         return tld_similarity
#     else:
#         return None

# def get_domain_name_similarity_score(url1, url2):
#     parts1 = extract_domain_parts(url1)
#     parts2 = extract_domain_parts(url2)
#     parts1.pop(-1)
#     parts2.pop(-1)
#     parts1 = ''.join(parts1)
#     parts2 = ''.join(parts2)

#     print(parts1,parts2,"domain")
#     if parts1 is not None and parts2 is not None:
#         domain_name_similarity = calculate_string_similarity(parts1, parts2)
#         return domain_name_similarity
#     else:
#         return None

# url1 = "https://icicirewards.online/"
# url2 = "https://www.icicibank.com/"

# tld_similarity_score = get_tld_similarity_score(url1, url2)
# domain_name_similarity_score = get_domain_name_similarity_score(url1, url2)

# print(f"TLD Similarity Score: {tld_similarity_score}")
# print(f"Domain Name Similarity Score: {domain_name_similarity_score}")

import re

class URLComparator:
    def __init__(self, url1, url2):
        self.url1 = url1
        self.url2 = url2

    def extract_domain_parts(self, url):
        match = re.search(r"https?://(?:www\.)?([^/]+)", url)
        if match:
            domain = match.group(1)
            parts = domain.split('.')
            return parts
        return None

    def calculate_string_similarity(self, str1, str2):
        m = len(str1)
        n = len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        max_len = max(len(str1), len(str2))
        similarity = 1 - (dp[m][n] / max_len)
        return similarity

    def get_tld_similarity_score(self):
        parts1 = self.extract_domain_parts(self.url1)
        parts2 = self.extract_domain_parts(self.url2)
        if parts1 is not None and parts2 is not None:
            tld_similarity = self.calculate_string_similarity(parts1[-1], parts2[-1])
            return tld_similarity
        else:
            return None

    def get_domain_name_similarity_score(self):
        parts1 = self.extract_domain_parts(self.url1)
        parts2 = self.extract_domain_parts(self.url2)
        parts1.pop(-1)
        parts2.pop(-1)
        parts1 = ''.join(parts1)
        parts2 = ''.join(parts2)

        if parts1 is not None and parts2 is not None:
            domain_name_similarity = self.calculate_string_similarity(parts1, parts2)
            return domain_name_similarity
        else:
            return None

# Example usage:
url1 = "https://icicirewards.online/"
url2 = "https://www.icicibank.com/"

url_comparator = URLComparator(url1, url2)
tld_similarity_score = url_comparator.get_tld_similarity_score()
domain_name_similarity_score = url_comparator.get_domain_name_similarity_score()

print(f"TLD Similarity Score: {tld_similarity_score}")
print(f"Domain Name Similarity Score: {domain_name_similarity_score}")
