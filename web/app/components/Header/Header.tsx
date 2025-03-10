import { AppBar, Tooltip, Container, Icon, IconButton, Toolbar, Typography } from "@mui/material";
import { GithubLogo, Fingerprint } from "@phosphor-icons/react";


export function Header() {
  return (
    <AppBar
      position="static"
      color="primary"
      sx={{
        boxShadow: "none",
      }}
    >
      <Container
        maxWidth="lg"
      >
        <Toolbar
          disableGutters
        >
          <Icon
            component={Fingerprint}
            sx={{
              mr: 1,
            }}
          />
          <Typography
            variant="h6"
            component="div"
          >
            Captcha Solver Demo
          </Typography>
          <Tooltip
            title="GitHub Repository"
          >
            <IconButton
              sx={{
                ml: "auto",
              }}
              href="https://github.com/Isaac-the-Man/captcha"
              target="_blank"
            >
              <Icon
                weight="fill"
                component={GithubLogo}
                sx={{
                  color: "white",
                }}
              />
            </IconButton>
          </Tooltip>
        </Toolbar>
      </Container>
    </AppBar>
  );
}