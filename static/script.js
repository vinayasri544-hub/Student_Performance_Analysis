
console.log("Student Performance Analysis loaded successfully.");

document.addEventListener("DOMContentLoaded", function () {
    let hamburger = document.querySelector(".hamburger");
    let sideMenu = document.getElementById("sideMenu");

    // Create hamburger if the page does not already have one
    if (!hamburger) {
        hamburger = document.createElement("button");
        hamburger.type = "button";
        hamburger.className = "hamburger";
        hamburger.setAttribute("aria-label", "Toggle navigation menu");
        hamburger.setAttribute("aria-expanded", "false");
        hamburger.innerHTML = "<span></span><span></span><span></span>";
        document.body.appendChild(hamburger);
    }

    // Create sidebar if the page does not already have one
    if (!sideMenu) {
        sideMenu = document.createElement("div");
        sideMenu.id = "sideMenu";
        sideMenu.className = "side-menu";

        sideMenu.innerHTML = `
            <h2>Project Menu</h2>
            <a href="/">Dashboard</a>
            <a href="/cleaning">Data Cleaning</a>
            <a href="/analysis">Analysis</a>
            <a href="/trends">Trends</a>
            <a href="/correlation">Correlation</a>
            <a href="/prediction">Prediction</a>
            <a href="/at-risk">At-Risk Students</a>
            <a href="/clustering">Clustering</a>
            <a href="/students">Students</a>
            <a href="/technical">Technical Knowledge</a>
            <a href="/coding-skills">Coding Skills</a>
            <a href="/placement-preparation">Placement Preparation</a>
            <a href="/placement-readiness">Placement Readiness Score</a>
            <a href="/report">Report & Recommendations</a>
        `;

        document.body.appendChild(sideMenu);
    }

    // Make the hamburger work on every page
    if (hamburger.dataset.menuReady !== "true") {
        hamburger.addEventListener("click", function () {
            const menu = document.getElementById("sideMenu");
            if (!menu) return;

            const isOpen = menu.classList.toggle("open");
            hamburger.setAttribute("aria-expanded", String(isOpen));
        });

        hamburger.dataset.menuReady = "true";
    }
});
