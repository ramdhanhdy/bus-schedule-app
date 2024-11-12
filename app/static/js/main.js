function updateTime() {
    const timeElement = document.getElementById('current-time');
    timeElement.textContent = new Date().toLocaleTimeString();
}

function updateSchedule() {
    fetch('/api/schedule/morning')
        .then(response => response.json())
        .then(data => {
            const tbody = document.querySelector('#morning-schedule tbody');
            tbody.innerHTML = '';
            
            data.schedule.forEach(stop => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${stop.name}</td>
                    <td>${stop.departure}</td>
                    <td>${stop.error_margin ? `±${stop.error_margin} min` : '-'}</td>
                `;
                tbody.appendChild(row);
            });
        })
        .catch(error => console.error('Error:', error));
}

// Update time every second
setInterval(updateTime, 1000);

// Update schedule every minute
updateSchedule();
setInterval(updateSchedule, 60000);