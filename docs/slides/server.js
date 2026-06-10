const http = require('http');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const PORT = 8080;
const HOST = 'localhost';
const BASE_DIR = path.join(__dirname);

const mimeTypes = {
    '.html': 'text/html',
    '.js': 'text/javascript',
    '.css': 'text/css',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
    '.json': 'application/json',
};

const server = http.createServer((req, res) => {
    let filePath = path.join(BASE_DIR, req.url === '/' ? 'index.html' : req.url);
    
    const extname = String(path.extname(filePath)).toLowerCase();
    const contentType = mimeTypes[extname] || 'application/octet-stream';

    fs.readFile(filePath, (error, content) => {
        if (error) {
            if (error.code === 'ENOENT') {
                res.writeHead(404, { 'Content-Type': 'text/html' });
                res.end('<h1>404 - File Not Found</h1>', 'utf-8');
            } else {
                res.writeHead(500);
                res.end('Server Error: ' + error.code);
            }
        } else {
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(content, 'utf-8');
        }
    });
});

server.on('error', (error) => {
    if (error.code === 'EADDRINUSE') {
        console.log(`端口 ${PORT} 已被占用，正在释放...`);
        try {
            execSync(`fuser -k ${PORT}/tcp 2>/dev/null || true`, { stdio: 'ignore' });
            console.log('已释放端口，1秒后重新启动...');
            setTimeout(() => {
                server.listen(PORT, HOST);
            }, 1000);
        } catch (e) {
            console.error(`无法释放端口 ${PORT}，请手动执行: fuser -k ${PORT}/tcp`);
            process.exit(1);
        }
    } else {
        throw error;
    }
});

server.listen(PORT, HOST, () => {
    console.log(`演示文稿服务器运行在 http://${HOST}:${PORT}`);
    console.log('按 Ctrl+C 停止服务器');
});

module.exports = server;