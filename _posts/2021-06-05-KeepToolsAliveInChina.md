---
layout: post
title:  "How to keep tools alive in China (incomplete)"
date:   2021-06-05 10:00:00
categories: Tooling
---

One trick of GFW is DNS pollution. We need to find the correct domain and ip address directly in host file explicitly.

## coursera

1. Go to https://www.ipaddress.com/
2. search `d3c33hcgiwev3.cloudfront.net`
3. `sudo vim /etc/hosts`

```
99.84.170.73 d3c33hcgiwev3.cloudfront.net
99.84.170.89 d3c33hcgiwev3.cloudfront.net
99.84.170.134 d3c33hcgiwev3.cloudfront.net
99.84.170.230 d3c33hcgiwev3.cloudfront.net
```

4. Restart `sudo /etc/init.d/networking restart`

## github

```
140.82.112.4 github.com
199.232.69.194 github.global.ssl.fastly.net
185.199.108.153 assets-cdn.github.com
185.199.110.153 assets-cdn.github.com
185.199.111.153 assets-cdn.github.com
```

## ladder

[v2ray](https://bella722.github.io/post/a231f91f.html)

[nodes](https://github.com/freefq/free)